/**
 * Experiment-matrix tool: turns one short matrix definition file into many experiment
 * .properties files plus the batch file that core.VascularBatchRunner runs. It is a separate
 * program (main class {@link matrix.ExperimentMatrixExpander}); the extraction code never uses it.
 * Dependency: matrix -> config (only to validate the generated files with the application's own parser).
 */
package matrix;
