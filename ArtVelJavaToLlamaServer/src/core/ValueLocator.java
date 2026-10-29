package core;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import core.VascularModels.ValueLocation;

/**
 * Finds the character offsets of an extracted value in the source document.
 *
 * The model returns, for each value, a short excerpt copied verbatim from the
 * report ("source_text") that contains the value. This class:
 *   1. finds every place the excerpt occurs in the document (exact match, then
 *      whitespace-insensitive, then case-insensitive);
 *   2. finds the value text inside each matched excerpt;
 *   3. if several candidate locations remain, picks one with a sequential
 *      heuristic and flags the result as ambiguous;
 *   4. takes CONTEXT_CHARS characters before and after the value directly from
 *      the DOCUMENT (never from the model), trimmed at the document boundaries.
 *
 * Use ONE ValueLocator PER SECTION PER DOCUMENT (ratios, velocities, stenosis)
 * and call locate() on that section's entries in the order the model returned
 * them. The instance keeps a cursor (start offset of the previously located
 * value) for the sequential heuristic: the model normally reports findings in
 * document order, so when the same excerpt/value occurs more than once, the
 * first occurrence AFTER the previous entry's value is chosen. If the model
 * does not report in document order, ambiguous results may be assigned to the
 * wrong occurrence, which is why they are flagged rather than hidden.
 *
 * Not thread-safe; instances are cheap and are created per document.
 */
public class ValueLocator {

    /** Characters of surrounding text captured before and after each value. */
    public static final int CONTEXT_CHARS = 25;

    private static final ValueLocation NOT_FOUND =
            new ValueLocation(false, -1, -1, "", "", "", false, "NOT_FOUND");

    private final String doc;
    private int lastValueStart = -1;

    public ValueLocator(String document) {
        this.doc = document;
    }

    private record Candidate(int start, int end, String method) {}

    /**
     * @param sourceText the model's verbatim excerpt containing the value (may be
     *                   null/blank or slightly inexact)
     * @param value      the extracted value (ratio_value, value, or percentage)
     */
    public ValueLocation locate(String sourceText, String value) {

        // Keyed by start offset so overlapping excerpt matches that resolve to
        // the same value occurrence count once.
        Map<Integer, Candidate> candidates = new LinkedHashMap<>();

        // --- Step 1/2: find the excerpt, then the value inside it ---
        if (!isBlank(sourceText)) {
            String method = "EXACT";
            List<int[]> spans = findAll(Pattern.compile(Pattern.quote(sourceText)));

            if (spans.isEmpty()) {
                String normalized = whitespaceTolerantRegex(sourceText);
                if (normalized != null) {
                    spans = findAll(Pattern.compile(normalized));
                    method = "WHITESPACE_NORMALIZED";
                    if (spans.isEmpty()) {
                        spans = findAll(Pattern.compile(normalized, Pattern.CASE_INSENSITIVE));
                        method = "CASE_INSENSITIVE";
                    }
                }
            }

            for (int[] span : spans) {
                List<int[]> valueSpans = findValue(value, span[0], span[1]);
                if (valueSpans.isEmpty()) {
                    // Excerpt found, value text not found inside it: report the
                    // excerpt's own span.
                    candidates.putIfAbsent(span[0],
                            new Candidate(span[0], span[1], method + "+SNIPPET_ONLY"));
                } else {
                    for (int[] vs : valueSpans) {
                        candidates.putIfAbsent(vs[0], new Candidate(vs[0], vs[1], method));
                    }
                }
            }
        }

        // --- Fallback: excerpt not found; search for the value across the doc ---
        if (candidates.isEmpty()) {
            for (int[] vs : findValue(value, 0, doc.length())) {
                candidates.putIfAbsent(vs[0], new Candidate(vs[0], vs[1], "VALUE_ONLY"));
            }
        }

        if (candidates.isEmpty()) {
            return NOT_FOUND;
        }

        // --- Step 3: choose a candidate ---
        List<Candidate> list = new ArrayList<>(candidates.values());
        Candidate chosen = list.stream()
                .filter(c -> c.start() > lastValueStart)
                .findFirst()
                .orElse(list.get(0));
        boolean ambiguous = list.size() > 1;
        lastValueStart = chosen.start();

        // --- Step 4: surrounding text from the document itself ---
        int s = chosen.start();
        int e = chosen.end();
        String before = doc.substring(Math.max(0, s - CONTEXT_CHARS), s);
        String matched = doc.substring(s, e);
        String after = doc.substring(e, Math.min(doc.length(), e + CONTEXT_CHARS));

        return new ValueLocation(true, s, e, before, matched, after, ambiguous, chosen.method());
    }

    // -------------------------------------------------------------------------

    private List<int[]> findAll(Pattern pattern) {
        List<int[]> spans = new ArrayList<>();
        Matcher m = pattern.matcher(doc);
        while (m.find()) {
            spans.add(new int[] { m.start(), m.end() });
        }
        return spans;
    }

    /**
     * Finds occurrences of the value text within doc[from, to). First tries the
     * value as written, bounded so a number is not matched inside a longer
     * number (e.g. "82" in "182" or "5" in "1.5"); if that finds nothing, tries
     * a version tolerant of whitespace around punctuation (e.g. "40 - 59 %").
     */
    private List<int[]> findValue(String value, int from, int to) {
        List<int[]> spans = new ArrayList<>();
        if (isBlank(value)) {
            return spans;
        }
        String v = value.strip();

        String strict = "(?<![0-9])(?<![0-9]\\.)" + Pattern.quote(v) + "(?![0-9])(?!\\.[0-9])";
        collect(Pattern.compile(strict), from, to, spans);

        if (spans.isEmpty()) {
            String loose = looseValueRegex(v);
            if (loose != null) {
                collect(Pattern.compile("(?<![0-9])(?<![0-9]\\.)" + loose + "(?![0-9])(?!\\.[0-9])"),
                        from, to, spans);
            }
        }
        return spans;
    }

    private void collect(Pattern p, int from, int to, List<int[]> out) {
        Matcher m = p.matcher(doc);
        // Transparent bounds let the lookbehind/lookahead see characters outside
        // the region so boundary checks are correct at the region's edges.
        m.region(from, to);
        m.useTransparentBounds(true);
        m.useAnchoringBounds(false);
        while (m.find()) {
            out.add(new int[] { m.start(), m.end() });
        }
    }

    // "a  b\nc" -> "\Qa\E\s+\Qb\E\s+\Qc\E"
    private static String whitespaceTolerantRegex(String s) {
        String trimmed = s.strip();
        if (trimmed.isEmpty()) {
            return null;
        }
        String[] tokens = trimmed.split("\\s+");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tokens.length; i++) {
            if (i > 0) {
                sb.append("\\s+");
            }
            sb.append(Pattern.quote(tokens[i]));
        }
        return sb.toString();
    }

    // Allows optional whitespace next to any non-alphanumeric character:
    // "40-59%" -> 4 0 \s*-\s* 5 9 \s*%   (digits/letters stay adjacent)
    private static String looseValueRegex(String v) {
        StringBuilder sb = new StringBuilder();
        char prev = 0;
        boolean first = true;
        for (char c : v.toCharArray()) {
            if (Character.isWhitespace(c)) {
                continue;
            }
            if (!first && (!Character.isLetterOrDigit(c) || !Character.isLetterOrDigit(prev))) {
                sb.append("\\s*");
            }
            sb.append(Pattern.quote(String.valueOf(c)));
            prev = c;
            first = false;
        }
        return first ? null : sb.toString();
    }

    private static boolean isBlank(String s) {
        return s == null || s.isBlank();
    }
}
