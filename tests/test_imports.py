
def test_import_backend():
    import backend.config as cfg
    assert hasattr(cfg, "CHAT_MODEL")
