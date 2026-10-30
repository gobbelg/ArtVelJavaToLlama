package core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.file.*;
import java.sql.*;

/**
 * Reads the JSON array file produced by MainVascularRevised and inserts its
 * contents into the SQL Server vascular schema using JDBC.
 *
 * This revision matches the XML-aligned JSON schema:
 *   - ICA_CCA_ratio   : laterality, ratio_value               (notation removed)
 *   - flow_velocities : vessel, laterality, vessel_location,
 *                       velocity_type, flow_direction, value, units
 *                                                             (flow_direction added; units
 *                                                              restored as cm/s|m/s|Other|Unspecified)
 *   - stenosis        : vessel, laterality, percentage        (additional_description removed)
 *   - plaque_and_intimal_thickening is no longer produced and is no longer imported.
 *
 * REQUIRED DATABASE CHANGES (run once before importing; adjust to your actual DDL):
 *
 *   -- New column required by this importer:
 *   ALTER TABLE ggLLM.FlowVelocity ADD FlowDirection NVARCHAR(50) NULL;
 *   -- Units is written again (values: cm/s, m/s, Other, Unspecified). If the column was
 *   -- dropped, re-add it; if it still exists, make sure it is wide enough:
 *   -- ALTER TABLE ggLLM.FlowVelocity ADD Units NVARCHAR(50) NULL;
 *
 *   -- Columns this importer no longer writes. They must either be dropped or be
 *   -- nullable / have a default, otherwise the INSERTs below will fail:
 *   --   ggLLM.IcaCcaRatio.Notation
 *   --   ggLLM.Stenosis.AdditionalDescription
 *   -- e.g.  ALTER TABLE ggLLM.IcaCcaRatio ALTER COLUMN Notation NVARCHAR(200) NULL;
 *   -- (ggLLM.PlaqueAndIntimalThickening is simply left unpopulated.)
 *
 * Required Maven / build dependency:
 *   com.microsoft.sqlserver:mssql-jdbc:12.6.1.jre11  (or latest)
 */
public class VascularDbImporter {

    // --- Connection settings — edit these ---
    private static final String DB_SERVER   = "VHACDWRB02.vha.med.va.gov";
    private static final String DB_NAME     = "ORD_Nguyen_202206062D";
//    private static final String DB_USER     = "";
//    private static final String DB_PASSWORD = "";

    private static final String JDBC_URL =
            "jdbc:sqlserver://" + DB_SERVER + ":1433;"
            + "databaseName=" + DB_NAME + ";"
//            + "user=" + DB_USER + ";"
//            + "password=" + DB_PASSWORD + ";"
//            + "encrypt=true;"
            + "integratedSecurity=true;"
            + "trustServerCertificate=true;";

    // --- SQL statements ---
    private static final String INSERT_DOCUMENT =
            "INSERT INTO ggLLM.VascularDocument (DocumentPath) " +
            "OUTPUT INSERTED.DocumentId " +
            "VALUES (?)";

    private static final String INSERT_RATIO =
            "INSERT INTO ggLLM.IcaCcaRatio (DocumentId, Laterality, RatioValue) " +
            "VALUES (?, ?, ?)";

    private static final String INSERT_VELOCITY =
            "INSERT INTO ggLLM.FlowVelocity " +
            "(DocumentId, Vessel, Laterality, VesselLocation, VelocityType, FlowDirection, Value, Units) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)";

    private static final String INSERT_STENOSIS =
            "INSERT INTO ggLLM.Stenosis " +
            "(DocumentId, Vessel, Laterality, Percentage) " +
            "VALUES (?, ?, ?, ?)";

    // -------------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: VascularDbImporterRevised <path-to-json-array-file>");
            System.exit(1);
        }

        Path jsonArrayPath = Paths.get(args[0]);
        if (!Files.exists(jsonArrayPath)) {
            System.err.println("File not found: " + jsonArrayPath.toAbsolutePath());
            System.exit(1);
        }

        importJsonArray(jsonArrayPath);
    }

    public static void importJsonArray(Path jsonArrayPath) throws IOException, SQLException {
        ObjectMapper mapper = new ObjectMapper();

        JsonNode root = mapper.readTree(jsonArrayPath.toFile());

        if (!root.isArray()) {
            System.err.println("Expected a JSON array at the root of: " + jsonArrayPath);
            System.exit(1);
        }

        System.out.println("Connecting to database...");

        try (Connection conn = DriverManager.getConnection(JDBC_URL)) {
            conn.setAutoCommit(false);  // one transaction per document

            int docCount = 0;
            int errorCount = 0;

            for (JsonNode doc : root) {
                String docPath = doc.path("documentPathString").asText();
                try {
                    importDocument(conn, doc, docPath);
                    conn.commit();
                    docCount++;
                    System.out.println("Imported: " + Paths.get(docPath).getFileName());
                } catch (Exception e) {
                    conn.rollback();
                    errorCount++;
                    System.err.println("Failed to import " + docPath + ": " + e.getMessage());
                }
            }

            System.out.println("Import complete. "
                    + docCount + " documents imported, "
                    + errorCount + " errors.");
        }
    }

    // -------------------------------------------------------------------------
    // Per-document import — all child inserts share the same transaction
    // -------------------------------------------------------------------------

    private static void importDocument(Connection conn, JsonNode doc, String docPath)
            throws SQLException {

        // 1. Insert document row and retrieve the generated DocumentId
        int documentId;
        try (PreparedStatement ps = conn.prepareStatement(INSERT_DOCUMENT)) {
            ps.setString(1, docPath);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) {
                    throw new SQLException("No DocumentId returned for: " + docPath);
                }
                documentId = rs.getInt(1);
            }
        }

        // 2. ICA/CCA ratios
        try (PreparedStatement ps = conn.prepareStatement(INSERT_RATIO)) {
            for (JsonNode r : doc.path("ICA_CCA_ratio")) {
                ps.setInt(1, documentId);
                ps.setString(2, r.path("laterality").asText("NoInformation"));
                ps.setString(3, r.path("ratio_value").asText("NoInformation"));
                ps.addBatch();
            }
            ps.executeBatch();
        }

        // 3. Flow velocities
        try (PreparedStatement ps = conn.prepareStatement(INSERT_VELOCITY)) {
            for (JsonNode v : doc.path("flow_velocities")) {
                ps.setInt(1, documentId);
                ps.setString(2, v.path("vessel").asText("NoInformation"));
                ps.setString(3, v.path("laterality").asText("NoInformation"));
                ps.setString(4, v.path("vessel_location").asText("NoInformation"));
                ps.setString(5, v.path("velocity_type").asText("NoInformation"));
                ps.setString(6, v.path("flow_direction").asText("NoInformation"));
                ps.setString(7, v.path("value").asText("NoInformation"));
                ps.setString(8, v.path("units").asText("NoInformation"));
                ps.addBatch();
            }
            ps.executeBatch();
        }

        // 4. Stenosis
        try (PreparedStatement ps = conn.prepareStatement(INSERT_STENOSIS)) {
            for (JsonNode s : doc.path("stenosis")) {
                ps.setInt(1, documentId);
                ps.setString(2, s.path("vessel").asText("NoInformation"));
                ps.setString(3, s.path("laterality").asText("NoInformation"));
                ps.setString(4, s.path("percentage").asText("NoInformation"));
                ps.addBatch();
            }
            ps.executeBatch();
        }
    }
}
