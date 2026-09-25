import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from cpu_process import CPUProcess, peak_rss_bytes, physical_cpu_ids, thread_environment
from make_quick_corpus import OUTPUT as QUICK_CORPUS, build as build_quick_corpus
from publish_issue7 import verify
from run_cpu_quick import CORPUS, SAMPLE_INDICES, sample_stratum


@unittest.skipUnless(sys.platform == 'linux', 'CPU process metrics use Linux /proc')
class CPUProcessTests(unittest.TestCase):
    def test_isolated_process_reports_readiness_affinity_and_peak_rss(self):
        cpus = physical_cpu_ids()
        self.assertTrue(cpus)
        program = (
            "import json,os,sys; "
            "print(json.dumps({'ready':True,'device':'cpu','threads':1}),flush=True); "
            "[(lambda x:print(json.dumps({'results':x,'elapsed_ms':1.0,'affinity':sorted(os.sched_getaffinity(0))}),flush=True))(json.loads(line)) for line in sys.stdin]"
        )
        with CPUProcess([sys.executable, '-c', program], threads=1, cpus=cpus[:1],
                        ready_stream='stdout', env=thread_environment(1)) as process:
            result = process.call([{'id': 'probe'}])
            self.assertEqual(result['results'], [{'id': 'probe'}])
            self.assertEqual(result['affinity'], cpus[:1])
            self.assertGreater(process.load_ms, 0)
            self.assertGreater(peak_rss_bytes(process.process.pid), 0)

    def test_thread_environment_sets_every_cpu_math_library(self):
        env = thread_environment(16)
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'LAYA_CPU_THREADS'):
            self.assertEqual(env[name], '16')

    def test_quick_selection_is_fixed_and_covers_every_output_type(self):
        for name, data in build_quick_corpus().items():
            self.assertEqual((QUICK_CORPUS / name).read_bytes(), data)
        manifest = json.loads(CORPUS.read_text())
        self.assertEqual(SAMPLE_INDICES, (0, 8, 16, 24, 32, 40, 48, 56))
        for item in manifest['strata']:
            _, cases, digest = sample_stratum(manifest, item['file'])
            self.assertEqual(len(cases), 8)
            self.assertEqual(len({case['id'] for case in cases}), 8)
            self.assertEqual(len(digest), 64)
            self.assertEqual({q['type'] for case in cases for q in case['questions'].values()},
                             {'choice', 'score', 'noul'})

    def test_cpu_process_call_obeys_a_short_deadline(self):
        cpus = physical_cpu_ids()
        program = ("import json,sys,time; "
                   "print(json.dumps({'ready':True,'device':'cpu','threads':1}),flush=True); "
                   "[(time.sleep(1),print('{}',flush=True)) for line in sys.stdin]")
        with self.assertRaises(TimeoutError):
            with CPUProcess([sys.executable, '-c', program], threads=1, cpus=cpus[:1],
                            ready_stream='stdout', env=thread_environment(1)) as process:
                process.call([{'id': 'probe'}], timeout=0.01)

    def test_publisher_rejects_an_incomplete_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run-index.json'
            path.write_text(json.dumps(dict(schema_version=1, kind='issue7-run-index',
                                            status='incomplete', corpus_id='performance-issue7-v1',
                                            corpus_manifest_sha256='0' * 64,
                                            candidate_sha256='0' * 64,
                                            candidate_commit='test', baseline_commit='test', entries=[])))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                verify(Path(directory))


if __name__ == '__main__':
    unittest.main()
