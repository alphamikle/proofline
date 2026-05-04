from proofline.utils import json_dumps


def test_json_dumps_converts_nested_dict_keys_to_strings():
    obj = {"responses": {200: {"headers": {"X-Test": {"schema": {"type": "string"}}}}}}

    assert json_dumps(obj) == '{"responses":{"200":{"headers":{"X-Test":{"schema":{"type":"string"}}}}}}'


def test_json_dumps_converts_tuple_values_to_arrays():
    assert json_dumps({"values": (1, 2)}) == '{"values":[1,2]}'
