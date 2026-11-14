# Refactor patterns

Authored by Christopher Wilson.

## Extract interface

For a file with many direct dependents, consider whether a small interface could
reduce callers' dependence on implementation details. A high blast score is a
reason to assess this option, not evidence that it is the right refactor or safe
to apply automatically.

1. Identify the contract callers actually use: inputs, outputs, errors, and side
   effects. Review the direct dependents in the impact report.
2. Preserve that behavior with regression tests before changing the contract.
   Review and run downstream test candidates, adding cases for uncovered behavior.
3. Introduce the smallest useful interface that hides implementation details.
   Avoid adding an abstraction when callers already depend on a clear contract.
4. Move callers to the interface in small steps, verifying affected callers and
   running their regression tests after each step. Keep observable behavior intact.
5. Re-run impact analysis to assess the resulting dependencies. An interface may
   clarify coupling without reducing the number of dependents or the blast score.

Test candidates in an impact report are inferred from conventional source paths
among direct and transitive dependents. Graph reachability does not establish test
coverage; tests using dynamic imports or indirect runtime behavior may be absent.
