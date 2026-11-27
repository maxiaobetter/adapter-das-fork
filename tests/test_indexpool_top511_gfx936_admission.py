"""Flash-only device admission; generic/DSV41 TopK keeps its old predicate."""
import ast
from pathlib import Path
from types import SimpleNamespace as N
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'hcu_adapter/kernels/indexer_topk_lightop.py'


class FlashTop511DeviceTests(unittest.TestCase):
    def test_qualified_93680_and_original_device_decision(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_indexpool_supported_device')
        for arch, cu, old, expected in [
            ('gfx936', 80, False, True), ('gfx936', 64, False, False),
            ('gfx938', 64, True, True), ('gfx942', 80, False, False),
        ]:
            with self.subTest(arch=arch, cu=cu):
                g = {'torch': N(cuda=N(get_device_properties=lambda _: N(
                    gcnArchName=arch + ':sramecc+', multi_processor_count=cu))),
                     '_supported_device': lambda _: old}
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-predicate>', 'exec'), g)
                self.assertEqual(g[node.name](0), expected)

    def test_only_flash_select_groups_uses_new_predicate(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        def named_calls(name):
            return [n.func.id for n in ast.walk(funcs[name])
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn('_indexpool_supported_device', named_calls('select_indexpool_groups'))
        self.assertNotIn('_indexpool_supported_device', named_calls('select_topk'))
        self.assertIn('_supported_device', named_calls('select_topk'))


if __name__ == '__main__':
    unittest.main()
