package core;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.function.Consumer;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Support for resuming a run: finds which documents are already present in an
 * existing JSON output file so they are not processed (and written) again.
 *
 * The JSON output file is a sequence of pretty-printed records separated by blank
 * lines, one per successfully processed document, each with a
 * "documentPathString" field. Only SUCCESSFUL documents are written to it, so a
 * document that failed or was skipped as too long is not "already processed" and
 * is picked up again by a resumed run.
 *
 * If the program was killed while writing, the last record may be incomplete.
 * That fragment is removed from the file (everything before it is untouched) so
 * that the records appended by the resumed run follow a complete one. The
 * fragment's document has no complete record, so it is processed again.
 *
 * Paths are compared after normalization (absolute, normalized, case-insensitive),
 * so "P:/data/a.txt" and "p:\data\a.txt" are the same document.
 */
public final class ResumeSupport {

    private ResumeSupport() {}

    /** Absolute, normalized, lower-cased form used to compare document paths. */
    public static String normalize(String path) {
        return Paths.get(path).toAbsolutePath().normalize().toString().toLowerCase(Locale.ROOT);
    }

    /**
     * Reads the documents already recorded in the JSON output file.
     *
     * @param log receives messages for the user (e.g. an incomplete last record)
     * @return the normalized paths (see normalize) of the documents already present
     */
    public static Set<String> loadProcessedPaths(Path jsonOutput, ObjectMapper mapper,
            Consumer<String> log) throws IOException {

        Set<String> done = new HashSet<>();
        long goodEnd = 0;       // byte offset just after the last complete record
        boolean incompleteTail = false;

        try (InputStream in = new BufferedInputStream(Files.newInputStream(jsonOutput));
                JsonParser parser = mapper.getFactory().createParser(in)) {

            MappingIterator<JsonNode> records = mapper.readValues(parser, JsonNode.class);
            while (true) {
                try {
                    if (!records.hasNextValue()) {
                        break;
                    }
                    JsonNode record = records.nextValue();
                    goodEnd = parser.getCurrentLocation().getByteOffset();

                    String path = record.path("documentPathString").asText("");
                    if (!path.isEmpty()) {
                        done.add(normalize(path));
                    }
                } catch (JsonProcessingException e) {
                    incompleteTail = true;
                    break;
                }
            }
        }

        if (incompleteTail) {
            long size = Files.size(jsonOutput);
            if (goodEnd > 0 && goodEnd < size) {
                try (FileChannel channel = FileChannel.open(jsonOutput, StandardOpenOption.WRITE)) {
                    channel.truncate(goodEnd);
                }
                log.accept("The last record in " + jsonOutput.getFileName() + " was incomplete ("
                        + (size - goodEnd) + " bytes, probably from an interrupted run); it was removed. "
                        + "That document will be processed again.");
            } else if (goodEnd == 0) {
                log.accept("WARNING: " + jsonOutput.getFileName() + " could not be read as JSON records; "
                        + "no documents will be skipped and the file was left unchanged.");
            }
        }
        return done;
    }
}
