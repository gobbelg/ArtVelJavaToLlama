package core;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import core.VascularModels.*;

/**
 * Writes the extraction results to three tab-delimited files, one per result
 * type, next to the JSON output file:
 *
 *   <base>_ICA_CCA_ratio.tsv
 *   <base>_flow_velocities.tsv
 *   <base>_stenosis.tsv
 *
 * where <base> is the JSON output file name without its extension. Each line
 * holds ONE extracted value from ONE document; the document's file name (not
 * its full path) is in the first column. start_offset / end_offset / matched_text
 * come from the value_location computed by ValueLocator (-1 / -1 / empty when
 * the value could not be located). Files are opened in append mode, matching the
 * JSON output; the header line is written only when a file is new or empty.
 *
 * Tabs, carriage returns, and line feeds inside any field are written as the
 * literal two-character escapes \t, \r, and \n so every record stays on one
 * line with the correct number of columns.
 *
 * NOT thread-safe: call write() from inside the caller's write lock (as
 * MainVascularRevised does).
 */
public class VascularTsvWriter implements AutoCloseable {

    // Column names. Edit here to rename columns.
    private static final List<String> RATIO_COLUMNS = List.of(
            "DocumentName", "laterality", "ratio_value",
            "start_offset", "end_offset", "matched_text");

    private static final List<String> VELOCITY_COLUMNS = List.of(
            "DocumentName", "vessel", "laterality", "vessel_location",
            "flow_direction", "velocity_type", "value",
            "start_offset", "end_offset", "matched_text");

    private static final List<String> STENOSIS_COLUMNS = List.of(
            "DocumentName", "vessel", "laterality", "percentage",
            "start_offset", "end_offset", "matched_text");

    private final PrintWriter ratioWriter;
    private final PrintWriter velocityWriter;
    private final PrintWriter stenosisWriter;

    private final Path ratioFile;
    private final Path velocityFile;
    private final Path stenosisFile;

    public VascularTsvWriter(Path jsonOutputFile) throws IOException {
        String name = jsonOutputFile.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        Path dir = jsonOutputFile.toAbsolutePath().getParent();

        ratioFile    = dir.resolve(base + "_ICA_CCA_ratio.tsv");
        velocityFile = dir.resolve(base + "_flow_velocities.tsv");
        stenosisFile = dir.resolve(base + "_stenosis.tsv");

        ratioWriter    = open(ratioFile, RATIO_COLUMNS);
        velocityWriter = open(velocityFile, VELOCITY_COLUMNS);
        stenosisWriter = open(stenosisFile, STENOSIS_COLUMNS);
    }

    public List<Path> getFiles() {
        return List.of(ratioFile, velocityFile, stenosisFile);
    }

    private static PrintWriter open(Path file, List<String> columns) throws IOException {
        // Decide about the header BEFORE opening, since append mode creates the file.
        boolean needsHeader = !Files.exists(file) || Files.size(file) == 0;

        PrintWriter writer = new PrintWriter(new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(file.toFile(), true), StandardCharsets.UTF_8)));
        if (needsHeader) {
            writer.print(String.join("\t", columns) + "\n");
            writer.flush();
        }
        return writer;
    }

    /** Writes one line per extracted value in the result, then flushes. */
    public void write(VascularResult result) {
        String docName = Paths.get(result.documentPathString()).getFileName().toString();

        for (IcaCcaRatio r : result.ICA_CCA_ratio()) {
            ValueLocation loc = r.value_location();
            ratioWriter.print(row(docName, r.laterality(), r.ratio_value(),
                    start(loc), end(loc), matched(loc)));
        }

        for (FlowVelocity v : result.flow_velocities()) {
            ValueLocation loc = v.value_location();
            velocityWriter.print(row(docName, v.vessel(), v.laterality(), v.vessel_location(),
                    v.flow_direction(), v.velocity_type(), v.value(),
                    start(loc), end(loc), matched(loc)));
        }

        for (Stenosis s : result.stenosis()) {
            ValueLocation loc = s.value_location();
            stenosisWriter.print(row(docName, s.vessel(), s.laterality(), s.percentage(),
                    start(loc), end(loc), matched(loc)));
        }

        ratioWriter.flush();
        velocityWriter.flush();
        stenosisWriter.flush();
    }

    @Override
    public void close() {
        ratioWriter.close();
        velocityWriter.close();
        stenosisWriter.close();
    }

    // -------------------------------------------------------------------------

    private static String start(ValueLocation loc) {
        return loc == null ? "-1" : Integer.toString(loc.start_offset());
    }

    private static String end(ValueLocation loc) {
        return loc == null ? "-1" : Integer.toString(loc.end_offset());
    }

    private static String matched(ValueLocation loc) {
        return loc == null ? "" : loc.matched_text();
    }

    private static String row(String... fields) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < fields.length; i++) {
            if (i > 0) {
                sb.append('\t');
            }
            sb.append(clean(fields[i]));
        }
        return sb.append('\n').toString();
    }

    // Keeps each record on one line with the right number of columns.
    private static String clean(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n");
    }
}
