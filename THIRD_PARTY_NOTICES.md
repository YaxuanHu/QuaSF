# Third-party materials

The root MIT license applies to original QuaSF code and documentation only.
Third-party source retains its own license and copyright notices:

- `code/baselines/chainsaw-main/`: Chainsaw by Jude Wells and contributors, MIT; see its LICENSE and https://github.com/JudeWells/Chainsaw . The `pdb_reres.py` file retains its Apache-2.0 notice.
- `code/baselines/merizo/`: Merizo by its original authors, GPL-3.0; see its LICENSE and https://github.com/psipred/Merizo . This is a separately executed baseline, not imported by the QuaSF package. The snapshot may include historical local adaptations and is not represented as an exact current upstream checkout.
- TED, UniProt sequences, Pfam/InterPro-to-GO annotations, GO ontology terms, AlphaFold structures, ESM2 and third-party model weights remain subject to their respective upstream terms. The root MIT license does not relicense these datasets or external model weights.

Benchmark scripts include local implementations/wrappers and saved historical outputs; method names alone do not certify equivalence to current upstream software. See the paper-to-code map and recorded provenance.
