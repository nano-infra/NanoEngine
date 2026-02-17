# NanoDeploy Documentation

This directory contains comprehensive documentation for the NanoDeploy project.

## Directory Structure

```
docs/
├── prefill-decode-disaggregation/    # Automatic peer discovery system
│   ├── README.md                      # Quick start guide
│   ├── architecture.md                # System architecture
│   ├── migration-guide.md             # Migration instructions
│   └── troubleshooting.md             # Common issues and fixes
│
├── serialization/                     # Serialization improvements
│   ├── pickle-migration.md            # Pickle to FlatBuffers migration
│   └── simplification.md              # Serialization simplification
│
├── engine_discovery_design.md         # Engine discovery design
├── README_ENGINE_DISCOVERY.md         # Engine discovery overview
├── CHANGELOG_DYNAMIC_DISCOVERY.md     # Discovery system changelog
├── examples.md                        # Example scripts usage guide
└── serialization_and_logging.md       # Serialization & logging notes
```

## Quick Links

### Prefill-Decode Disaggregation

- [Getting Started](prefill-decode-disaggregation/README.md)
- [Architecture Overview](prefill-decode-disaggregation/architecture.md)
- [Migration Guide](prefill-decode-disaggregation/migration-guide.md)
- [Troubleshooting](prefill-decode-disaggregation/troubleshooting.md)

### Serialization

- [Pickle Migration](serialization/pickle-migration.md)
- [Simplification](serialization/simplification.md)

### Example Scripts

- [Example Scripts Guide](examples.md) — Usage for `non_disagg.py` and `disagg.py`

### Engine Discovery

- [Engine Discovery Design](engine_discovery_design.md)
- [Engine Discovery Overview](README_ENGINE_DISCOVERY.md)

## Contributing

When adding new documentation:

1. Place it in the appropriate subdirectory
2. Update this README with a link
3. Use clear, descriptive filenames (lowercase with hyphens)
4. Include a brief description in this index
