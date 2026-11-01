package model;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;

import java.io.IOException;

/**
 * Deserializes a JSON string field, replacing both JSON null and a missing
 * value with "NoInformation" so that downstream validation never receives null.
 */
public class NullToNoInformationDeserializer extends StdDeserializer<String> {

    private static final long serialVersionUID = -8957710393153803695L;
	public static final String NO_INFORMATION = "NoInformation";

    public NullToNoInformationDeserializer() {
        super(String.class);
    }

    @Override
    public String deserialize(JsonParser p, DeserializationContext ctx) throws IOException {
        String value = p.getValueAsString();
        return (value == null || value.isBlank()) ? NO_INFORMATION : value;
    }

    @Override
    public String getNullValue(DeserializationContext ctx) {
        // Called when the JSON token is literally null
        return NO_INFORMATION;
    }
}
