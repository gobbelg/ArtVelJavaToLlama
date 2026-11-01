package model;

import java.util.List;

/**
 * Revised data model aligned with the XML schema used for manually extracted
 * values. Differences from core.vascular.VascularModels:
 *   - ICA_CCA_ratio       : "notation" removed; "source_text" added.
 *   - flow_velocities     : "flow_direction", "source_text" and "units" (cm/s | m/s | Other | Unspecified) added.
 *   - stenosis            : "additional_description" removed; "source_text" added.
 *   - plaque_and_intimal_thickening section removed entirely.
 *
 * "source_text" is a short excerpt the model copies VERBATIM from the report and
 * which contains the extracted value. Java post-processing (ValueLocator) uses
 * it to find the value's character offsets in the document and attaches the
 * result to each enriched record as "value_location".
 *
 * This is a NEW class (rather than an in-place edit of VascularModels) so that
 * the original extractor/importer code that still depends on the old records
 * continues to compile.
 */
public class VascularModels {

    // -------------------------------------------------------------------------
    // Model-layer records (raw LLM output, no document metadata)
    // -------------------------------------------------------------------------

    public record ModelIcaCcaRatio(
            String laterality,
            String source_text,
            String ratio_value
    ) {}

    public record ModelFlowVelocity(
            String vessel,
            String laterality,
            String vessel_location,
            String velocity_type,
            String flow_direction,
            String source_text,
            String value,
            String units
    ) {}

    public record ModelStenosis(
            String vessel,
            String laterality,
            String source_text,
            String percentage
    ) {}

    /** Root model result — mirrors the JSON schema returned by the LLM. */
    public record ModelVascularResult(
            List<ModelIcaCcaRatio>  ICA_CCA_ratio,
            List<ModelFlowVelocity> flow_velocities,
            List<ModelStenosis>     stenosis
    ) {}


    // -------------------------------------------------------------------------
    // Location of an extracted value within the source document
    // -------------------------------------------------------------------------

    /**
     * Where an extracted value was found in the document.
     *
     * @param found           false if the value could not be located (all other
     *                        fields are then -1 / empty / "NOT_FOUND")
     * @param start_offset    index of the first character of the value text, as a
     *                        Java String (UTF-16 char) offset into the text read
     *                        by LlamaVascularExtractor.readDocumentText (line endings preserved; BOM excluded)
     * @param end_offset      index one past the last character (exclusive), so
     *                        document.substring(start_offset, end_offset) is the
     *                        value text
     * @param text_before     up to ValueLocator.CONTEXT_CHARS characters before
     *                        the value (fewer at the start of the document)
     * @param matched_text    the document text at [start_offset, end_offset)
     * @param text_after      up to ValueLocator.CONTEXT_CHARS characters after
     *                        the value (fewer at the end of the document)
     * @param ambiguous       true if more than one place in the document matched
     *                        the model's excerpt/value, so the chosen location is
     *                        a best guess (see ValueLocator for the heuristic)
     * @param match_method    how the location was found: EXACT,
     *                        WHITESPACE_NORMALIZED, CASE_INSENSITIVE (excerpt
     *                        matched), VALUE_ONLY (excerpt not found; value
     *                        searched across the document), or NOT_FOUND. A
     *                        "+SNIPPET_ONLY" suffix means the excerpt was found
     *                        but the value text was not found inside it, so the
     *                        offsets cover the whole excerpt.
     */
    public record ValueLocation(
            boolean found,
            int     start_offset,
            int     end_offset,
            String  text_before,
            String  matched_text,
            String  text_after,
            boolean ambiguous,
            String  match_method
    ) {}


    // -------------------------------------------------------------------------
    // Enriched records (model values + document metadata added post-extraction)
    // -------------------------------------------------------------------------

    public record IcaCcaRatio(
            String laterality,
            String source_text,
            String ratio_value,
            ValueLocation value_location
    ) {}

    public record FlowVelocity(
            String vessel,
            String laterality,
            String vessel_location,
            String velocity_type,
            String flow_direction,
            String source_text,
            String value,
            String units,
            ValueLocation value_location
    ) {}

    public record Stenosis(
            String vessel,
            String laterality,
            String source_text,
            String percentage,
            ValueLocation value_location
    ) {}

    /** Root enriched result returned to callers. */
    public record VascularResult(
            List<IcaCcaRatio>  ICA_CCA_ratio,
            List<FlowVelocity> flow_velocities,
            List<Stenosis>     stenosis,
            String             documentPathString
    ) {}
}
