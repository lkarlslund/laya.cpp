#include "laya/coreml.hpp"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace laya {
namespace {

json read_json(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open " + path.string());
    return json::parse(file);
}

std::string ns_string(NSString* value) {
    if (!value) return {};
    const char* text = value.UTF8String;
    return text ? std::string(text) : std::string();
}

std::string ns_error(NSError* error) {
    return error ? ns_string(error.localizedDescription) : "unknown Core ML error";
}

[[noreturn]] void incompatible(const std::string& detail) {
    throw std::runtime_error("Incompatible Core ML model: " + detail);
}

void require_multi_array(NSDictionary<NSString*, MLFeatureDescription*>* descriptions, NSString* name,
                         MLMultiArrayDataType type, const char* role) {
    MLFeatureDescription* description = [descriptions objectForKey:name];
    if (!description) incompatible(std::string("missing ") + role + " '" + ns_string(name) + "'");
    if (description.type != MLFeatureTypeMultiArray)
        incompatible(std::string(role) + " '" + ns_string(name) + "' is not a multi-array");
    MLMultiArrayConstraint* constraint = description.multiArrayConstraint;
    if (constraint && constraint.dataType != type)
        incompatible(std::string(role) + " '" + ns_string(name) + "' has an unexpected element type");
}

MLMultiArray* make_input(NSArray<NSNumber*>* shape, const int32_t* source, size_t count, const char* name) {
    NSError* error = nil;
    MLMultiArray* value = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeInt32 error:&error];
    if (!value) throw std::runtime_error("Cannot allocate Core ML input '" + std::string(name) + "': " + ns_error(error));
    if (value.count != static_cast<NSInteger>(count)) {
        [value release];
        throw std::runtime_error("Core ML input '" + std::string(name) + "' has an unexpected element count");
    }
    std::memcpy(value.dataPointer, source, count * sizeof(int32_t));
    return [value autorelease];
}

size_t checked_size(int first, int second, const char* name);

void copy_output(MLMultiArray* value, const char* name, int rows, int columns, std::vector<float>& destination) {
    if (!value) incompatible(std::string("missing output '") + name + "'");
    if (value.shape.count != 2 || value.shape[0].intValue != rows || value.shape[1].intValue != columns)
        incompatible(std::string("output '") + name + "' does not have shape [batch, columns]");
    if (value.dataType != MLMultiArrayDataTypeFloat32)
        incompatible(std::string("output '") + name + "' is not Float32");
    const NSInteger row_stride = value.strides[0].integerValue;
    const NSInteger column_stride = value.strides[1].integerValue;
    const auto* source = static_cast<const float*>(value.dataPointer);
    destination.resize(checked_size(rows, columns, "output"));
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column)
            destination[static_cast<size_t>(row) * columns + column] = source[row * row_stride + column * column_stride];
}

size_t checked_size(int first, int second, const char* name) {
    if (first < 1 || second < 1 || static_cast<size_t>(first) > std::numeric_limits<size_t>::max() / static_cast<size_t>(second))
        throw std::runtime_error(std::string("Invalid Core ML ") + name + " dimensions");
    return static_cast<size_t>(first) * static_cast<size_t>(second);
}

int manifest_dimension(const json& bucket, const char* key) {
    if (!bucket.contains(key) || !bucket.at(key).is_number_integer())
        incompatible(std::string("bucket is missing integer '") + key + "'");
    const auto value = bucket.at(key).get<int64_t>();
    if (value < 1 || value > std::numeric_limits<int>::max())
        incompatible(std::string("bucket has invalid '") + key + "'");
    return static_cast<int>(value);
}

}  // namespace

struct coreml_runtime::impl {
    struct bucket {
        int batch = 0;
        int length = 0;
        int options = 0;
        std::filesystem::path compiled;
        std::string name;
    };

    std::filesystem::path directory;
    json model_config;
    int vocabulary = 0;
    int action_count = 0;
    std::vector<bucket> buckets;
    std::map<std::string, MLModel*> models;

    explicit impl(const std::filesystem::path& model_directory) : directory(model_directory) {
        model_config = read_json(directory / "rl_agent_config.json");
        const auto encoder = read_json(directory / "encoder/config.json");
        vocabulary = encoder.value("vocab_size", 0);
        if (vocabulary < 1) throw std::runtime_error("Invalid encoder vocabulary for Core ML model");
        if (!model_config.contains("act_costs") ||
            (!model_config.at("act_costs").is_array() && !model_config.at("act_costs").is_object()))
            throw std::runtime_error("Invalid rl_agent_config.json: missing act_costs for Core ML model");
        if (model_config.at("act_costs").size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Invalid rl_agent_config.json: too many Core ML actions");
        action_count = static_cast<int>(model_config.at("act_costs").size()) + 1;

        const auto manifest_path = directory / "coreml/manifest.json";
        if (!std::filesystem::is_regular_file(manifest_path))
            throw std::runtime_error("Core ML manifest missing: " + manifest_path.string());
        const auto manifest = read_json(manifest_path);
        if (!manifest.is_object() || manifest.value("schema_version", 0) != 1)
            throw std::runtime_error("Unsupported Core ML manifest schema; expected schema_version=1: " + manifest_path.string());
        if (!manifest.contains("buckets") || !manifest.at("buckets").is_array() || manifest.at("buckets").empty())
            throw std::runtime_error("Incompatible Core ML manifest: missing nonempty buckets array: " + manifest_path.string());
        for (const auto& entry : manifest.at("buckets")) {
            if (!entry.is_object()) incompatible("bucket entry is not an object");
            bucket parsed;
            parsed.batch = manifest_dimension(entry, "batch");
            parsed.length = manifest_dimension(entry, "length");
            parsed.options = manifest_dimension(entry, "options");
            if (parsed.options < 2 || parsed.options > 255)
                incompatible("bucket options must be between 2 and 255");
            if (parsed.length > model_config.value("max_len", 512))
                incompatible("bucket length exceeds checkpoint max_len");
            if (!entry.contains("name") || !entry.at("name").is_string() || !entry.contains("compiled") || !entry.at("compiled").is_string())
                incompatible("bucket is missing name or compiled artifact path");
            const auto expected_name = "b" + std::to_string(parsed.batch) + "-l" + std::to_string(parsed.length) + "-o" + std::to_string(parsed.options);
            const auto expected_compiled = expected_name + "/Laya.mlmodelc";
            if (entry.at("name").get<std::string>() != expected_name || entry.at("compiled").get<std::string>() != expected_compiled)
                incompatible("bucket metadata does not match the required b{B}-l{L}-o{O}/Laya.mlmodelc layout");
            checked_size(parsed.batch, parsed.length, "bucket");
            checked_size(parsed.batch, parsed.options, "bucket");
            parsed.name = expected_name;
            parsed.compiled = directory / "coreml" / expected_compiled;
            buckets.push_back(std::move(parsed));
        }
    }

    ~impl() {
        for (auto& entry : models) [entry.second release];
    }

    void validate_batch(const batch& input) const {
        if (input.size < 1 || input.length < 1 || input.length > model_config.value("max_len", 512) || input.options < 2 || input.options > 255 ||
            input.ids.size() != checked_size(input.size, input.length, "batch"))
            throw std::runtime_error("Invalid Core ML model batch");
        if (input.lengths.size() != static_cast<size_t>(input.size) || input.types.size() != static_cast<size_t>(input.size) ||
            input.counts.size() != static_cast<size_t>(input.size) || input.markers.size() != checked_size(input.size, input.options, "batch"))
            throw std::runtime_error("Invalid Core ML batch metadata sizes");
        for (int row = 0; row < input.size; ++row) {
            if (input.lengths[row] < 1 || input.lengths[row] > input.length || input.types[row] < 0 || input.types[row] > 2 ||
                input.counts[row] < 2 || input.counts[row] > input.options)
                throw std::runtime_error("Invalid Core ML batch row metadata");
            for (int column = 0; column < input.counts[row]; ++column) {
                const auto marker = static_cast<int64_t>(input.markers[static_cast<size_t>(row) * input.options + column]) -
                                    static_cast<int64_t>(row) * input.length;
                if (marker < 0 || marker >= input.lengths[row])
                    throw std::runtime_error("Core ML option marker outside its sequence");
            }
        }
        for (const auto id : input.ids)
            if (id < 0 || id >= vocabulary) throw std::runtime_error("Core ML token ID outside the vocabulary");
    }

    const bucket& bucket_for(const batch& input) const {
        const bucket* selected = nullptr;
        for (const auto& candidate : buckets) {
            if (candidate.batch < input.size || candidate.length < input.length || candidate.options < input.options) continue;
            if (!selected || std::tie(candidate.batch, candidate.length, candidate.options) <
                                 std::tie(selected->batch, selected->length, selected->options))
                selected = &candidate;
        }
        if (selected) return *selected;
        std::string available;
        for (const auto& candidate : buckets) {
            if (!available.empty()) available += ", ";
            available += candidate.name;
        }
        throw std::runtime_error("Core ML bucket unavailable for batch=" + std::to_string(input.size) +
                                 ", length=" + std::to_string(input.length) + ", options=" + std::to_string(input.options) +
                                 "; need a bucket whose batch/length/options are at least this large. Available: " + available);
    }

    batch pad_for_bucket(const batch& input, const bucket& selected) const {
        batch padded;
        padded.size = selected.batch;
        padded.length = selected.length;
        padded.options = selected.options;
        padded.ids.assign(checked_size(padded.size, padded.length, "padded batch"), 0);
        padded.markers.assign(checked_size(padded.size, padded.options, "padded batch"), 0);
        padded.lengths.assign(padded.size, std::min(3, padded.length));
        padded.counts.assign(padded.size, 2);
        padded.types.assign(padded.size, 0);
        std::copy(input.lengths.begin(), input.lengths.end(), padded.lengths.begin());
        std::copy(input.counts.begin(), input.counts.end(), padded.counts.begin());
        std::copy(input.types.begin(), input.types.end(), padded.types.begin());
        for (int row = 0; row < padded.size; ++row) {
            const auto offset = static_cast<int64_t>(row) * padded.length;
            for (int column = 0; column < padded.options; ++column) {
                const auto local = std::min(column + 1, padded.length - 1);
                padded.markers[static_cast<size_t>(row) * padded.options + column] =
                    static_cast<int32_t>(offset + local);
            }
        }
        for (int row = 0; row < input.size; ++row) {
            std::copy_n(input.ids.begin() + static_cast<size_t>(row) * input.length, input.length,
                        padded.ids.begin() + static_cast<size_t>(row) * padded.length);
            for (int column = 0; column < input.counts[row]; ++column) {
                const auto old_marker = input.markers[static_cast<size_t>(row) * input.options + column];
                const auto local_marker = static_cast<int64_t>(old_marker) - static_cast<int64_t>(row) * input.length;
                const auto new_marker = static_cast<int64_t>(row) * padded.length + local_marker;
                if (local_marker < 0 || local_marker >= input.length || new_marker > std::numeric_limits<int32_t>::max())
                    throw std::runtime_error("Cannot pad Core ML marker index safely");
                padded.markers[static_cast<size_t>(row) * padded.options + column] = static_cast<int32_t>(new_marker);
            }
        }
        return padded;
    }

    MLModel* model_for(const bucket& selected) {
        const auto& path = selected.compiled;
        if (!std::filesystem::is_directory(path))
            throw std::runtime_error("Core ML compiled bucket is missing: " + path.string());
        const auto key = path.string();
        if (const auto found = models.find(key); found != models.end()) return found->second;

        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:key.c_str()]];
        MLModelConfiguration* configuration = [[MLModelConfiguration alloc] init];
        configuration.computeUnits = MLComputeUnitsAll;
        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:configuration error:&error];
        [configuration release];
        if (!model) throw std::runtime_error("Cannot load Core ML model " + key + ": " + ns_error(error));

        const auto inputs = model.modelDescription.inputDescriptionsByName;
        for (NSString* name in @[ @"ids", @"lengths", @"markers", @"counts", @"types" ])
            require_multi_array(inputs, name, MLMultiArrayDataTypeInt32, "input");
        const auto outputs = model.modelDescription.outputDescriptionsByName;
        require_multi_array(outputs, @"logits", MLMultiArrayDataTypeFloat32, "output");
        require_multi_array(outputs, @"actions", MLMultiArrayDataTypeFloat32, "output");

        [model retain];
        return models.emplace(key, model).first->second;
    }

    raw_result forward(const batch& input) {
        validate_batch(input);
        const auto& selected = bucket_for(input);
        const auto padded = pad_for_bucket(input, selected);
        @autoreleasepool {
            MLModel* model = model_for(selected);
            NSError* error = nil;
            MLMultiArray* ids = make_input(@[ @(padded.size), @(padded.length) ], padded.ids.data(), padded.ids.size(), "ids");
            MLMultiArray* lengths = make_input(@[ @(padded.size) ], padded.lengths.data(), padded.lengths.size(), "lengths");
            MLMultiArray* markers = make_input(@[ @(padded.size), @(padded.options) ], padded.markers.data(), padded.markers.size(), "markers");
            MLMultiArray* counts = make_input(@[ @(padded.size) ], padded.counts.data(), padded.counts.size(), "counts");
            MLMultiArray* types = make_input(@[ @(padded.size) ], padded.types.data(), padded.types.size(), "types");
            NSDictionary* features = @{
                @"ids": [MLFeatureValue featureValueWithMultiArray:ids],
                @"lengths": [MLFeatureValue featureValueWithMultiArray:lengths],
                @"markers": [MLFeatureValue featureValueWithMultiArray:markers],
                @"counts": [MLFeatureValue featureValueWithMultiArray:counts],
                @"types": [MLFeatureValue featureValueWithMultiArray:types],
            };
            MLDictionaryFeatureProvider* provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
            if (!provider) throw std::runtime_error("Cannot construct Core ML feature provider: " + ns_error(error));

            const auto started = std::chrono::steady_clock::now();
            id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
            [provider release];
            if (!prediction) throw std::runtime_error("Core ML prediction failed: " + ns_error(error));

            raw_result result;
            result.action_count = action_count;
            std::vector<float> padded_logits;
            copy_output([[prediction featureValueForName:@"logits"] multiArrayValue], "logits", padded.size, padded.options, padded_logits);
            result.logits.resize(checked_size(input.size, input.options, "result"));
            for (int row = 0; row < input.size; ++row)
                std::copy_n(padded_logits.begin() + static_cast<size_t>(row) * padded.options, input.options,
                            result.logits.begin() + static_cast<size_t>(row) * input.options);
            std::vector<float> padded_actions;
            copy_output([[prediction featureValueForName:@"actions"] multiArrayValue], "actions", padded.size, action_count, padded_actions);
            result.actions.assign(padded_actions.begin(), padded_actions.begin() + checked_size(input.size, action_count, "result"));
            result.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            return result;
        }
    }
};

coreml_runtime::coreml_runtime(const std::filesystem::path& directory) : p(std::make_unique<impl>(directory)) {}
coreml_runtime::~coreml_runtime() = default;
raw_result coreml_runtime::forward(const batch& input) { return p->forward(input); }
const json& coreml_runtime::config() const { return p->model_config; }
std::string coreml_runtime::backend_name() const { return "coreml"; }
std::string coreml_runtime::device_name() const { return "Core ML (all compute units)"; }

}  // namespace laya
