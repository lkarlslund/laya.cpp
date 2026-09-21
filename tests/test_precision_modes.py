import contextlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
from native import Native


class NativePrecisionTests(unittest.TestCase):
    def test_each_precision_reaches_the_process_explicitly(self):
        for fp32,fp16,flag in [(True,False,'--fp32'),(False,False,'--experimental-bf16'),(False,True,'--fp16')]:
            with self.subTest(flag=flag), patch('native.subprocess.Popen') as process:
                Native('laya-cli','model',backend='vulkan',fp32=fp32,fp16=fp16,flash=True)
                command=process.call_args.args[0]
                self.assertIn(flag,command)
                self.assertIn('--vulkan',command)
                self.assertEqual(sum(arg in command for arg in ['--fp32','--fp16','--experimental-bf16']),1)

    def test_conflicting_precision_never_starts_process(self):
        with patch('native.subprocess.Popen') as process:
            with self.assertRaises(ValueError): Native('laya-cli','model',fp32=True,fp16=True)
            process.assert_not_called()


class OraclePrecisionTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch comparison tools are not installed')
        self.torch=torch
        self.old_path=sys.path.copy()
        self.agent=SimpleNamespace(dtype=torch.bfloat16,device=SimpleNamespace(type='cuda'),model=lambda *args:'output')
        package=ModuleType('laya');package.load=lambda *args,**kwargs:self.agent
        common=ModuleType('laya.common')
        common.build_sequence=lambda *args:None
        common.collate_items=lambda *args:None
        common.QTYPES={}
        self.modules=patch.dict(sys.modules,{'laya':package,'laya.common':common})
        self.modules.start()
        from oracle import Oracle
        self.Oracle=Oracle

    def tearDown(self):
        if hasattr(self,'modules'):
            self.modules.stop();sys.path[:]=self.old_path

    def test_fp16_selects_baseline_autocast_not_bf16(self):
        oracle=self.Oracle('unused','unused',fp16=True)
        self.assertEqual(self.agent.dtype,self.torch.float16)
        with patch.object(self.torch,'autocast',return_value=contextlib.nullcontext()) as autocast:
            self.assertEqual(oracle.forward(()),'output')
            autocast.assert_called_once_with('cuda',dtype=self.torch.float16,enabled=True)

    def test_bf16_does_not_accept_silent_fp32_fallback(self):
        self.agent.dtype=self.torch.float32
        with self.assertRaisesRegex(RuntimeError,'different autocast dtype'):
            self.Oracle('unused','unused')


if __name__=='__main__': unittest.main()
