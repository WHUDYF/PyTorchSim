# Route 4 fusion-only pilot correctness blocker

Workload: `addmm_relu_128`

Correctness gate failed or did not complete for at least one fusion variant.

## fuse_none

- allclose_passed: `False`
- child_returncode: `2`
- error: `spike: unrecognized option --varch=vlen:256,elen:64`

## fuse_all

- allclose_passed: `False`
- child_returncode: `2`
- error: `spike: unrecognized option --varch=vlen:256,elen:64`
