if __name__ == "typing":
    # Imported as `import typing`, not `import <pckg>.typing`
    # Remove from sys.modules and replace with builtin `parameters`
    import sys
    from mackelab_toolbox.meta import HideCWDFromImport
    del sys.modules['typing']
    with HideCWDFromImport(__file__):
        import typing
else:
    from .transform import transformed_frozen, mvtransformed_frozen, mvjoint_frozen

    stats_encoders = {}
    stats_encoders[transformed_frozen] = transformed_frozen.json_encoder
    stats_encoders[mvtransformed_frozen] = mvtransformed_frozen.json_encoder
    stats_encoders[mvjoint_frozen] = mvjoint_frozen.json_encoder
