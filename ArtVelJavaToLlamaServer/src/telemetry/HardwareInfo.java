package telemetry;

import java.io.File;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Best-effort description of the computer, recorded in each run's manifest so
 * results from different machines can be compared later. Every lookup is wrapped:
 * a failure gives an empty or "unknown" value, never an exception.
 */
public final class HardwareInfo {

    private HardwareInfo() {
    }

    public static String hostName() {
        String n = System.getenv("COMPUTERNAME");
        if (n == null || n.isEmpty()) {
            n = System.getenv("HOSTNAME");
        }
        if (n == null || n.isEmpty()) {
            try {
                n = InetAddress.getLocalHost().getHostName();
            } catch (Exception e) {
                n = "unknown-host";
            }
        }
        return n;
    }

    public static Map<String, Object> describe() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("host_name", hostName());
        m.put("os", System.getProperty("os.name") + " " + System.getProperty("os.version")
                + " (" + System.getProperty("os.arch") + ")");
        m.put("cpu", cpuName());
        m.put("logical_processors", Runtime.getRuntime().availableProcessors());
        m.put("physical_memory_gb", physicalMemoryGb());
        m.put("gpu", gpus());
        m.put("java", System.getProperty("java.version") + " (" + System.getProperty("java.vendor") + ")");
        return m;
    }

    private static String cpuName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        try {
            if (os.contains("win")) {
                String p = System.getenv("PROCESSOR_IDENTIFIER");
                return p == null ? "unknown" : p;
            }
            if (os.contains("mac")) {
                String s = run(5, "sysctl", "-n", "machdep.cpu.brand_string").trim();
                return s.isEmpty() ? "unknown" : s;
            }
            Path info = new File("/proc/cpuinfo").toPath();
            if (Files.isRegularFile(info)) {
                for (String line : Files.readAllLines(info, StandardCharsets.UTF_8)) {
                    if (line.startsWith("model name")) {
                        return line.substring(line.indexOf(':') + 1).trim();
                    }
                }
            }
        } catch (Exception e) {
            // fall through
        }
        return "unknown";
    }

    private static Object physicalMemoryGb() {
        try {
            java.lang.management.OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
            // Reflection: the method is getTotalMemorySize() on Java 14+ and
            // getTotalPhysicalMemorySize() (deprecated later) before that.
            for (String name : new String[] {"getTotalMemorySize", "getTotalPhysicalMemorySize"}) {
                try {
                    java.lang.reflect.Method m = os.getClass().getMethod(name);
                    m.setAccessible(true);
                    long bytes = ((Number) m.invoke(os)).longValue();
                    return Math.round(bytes / (1024.0 * 1024 * 1024) * 10) / 10.0;
                } catch (Exception e) {
                    // try the next name
                }
            }
        } catch (Throwable e) {
            // fall through
        }
        return "unknown";
    }

    private static List<String> gpus() {
        List<String> out = new ArrayList<>();
        String os = System.getProperty("os.name", "").toLowerCase();
        try {
            String s = run(10, "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader");
            for (String line : s.split("\\r?\\n")) {
                if (!line.trim().isEmpty() && !line.toLowerCase().contains("not found")) {
                    out.add("NVIDIA " + line.trim());
                }
            }
        } catch (Exception e) {
            // no nvidia-smi
        }
        if (out.isEmpty() && os.contains("mac")) {
            try {
                String s = run(15, "system_profiler", "SPDisplaysDataType");
                for (String line : s.split("\\r?\\n")) {
                    String t = line.trim();
                    if (t.startsWith("Chipset Model:") || t.startsWith("Total Number of Cores:")
                            || t.startsWith("Metal")) {
                        out.add(t);
                    }
                }
            } catch (Exception e) {
                // ignore
            }
        }
        if (out.isEmpty()) {
            out.add("none detected");
        }
        return out;
    }

    /** First lines of "llama-server --version" (build number, backend), or a note on failure. */
    public static String llamaVersion(Path executable) {
        try {
            String s = run(20, executable.toString(), "--version").trim();
            return s.isEmpty() ? "unknown" : s;
        } catch (Exception e) {
            return "unknown (" + e.getClass().getSimpleName() + ")";
        }
    }

    /** Runs a command with output (stdout+stderr) to a temp file, a timeout and forced cleanup. */
    private static String run(int timeoutSeconds, String... command) throws IOException, InterruptedException {
        Path tmp = Files.createTempFile("vascular-hw", ".txt");
        Process p = null;
        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            pb.redirectOutput(tmp.toFile());
            p = pb.start();
            if (!p.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                throw new IOException("timed out: " + command[0]);
            }
            return new String(Files.readAllBytes(tmp), StandardCharsets.UTF_8);
        } finally {
            if (p != null && p.isAlive()) {
                p.destroyForcibly();
            }
            Files.deleteIfExists(tmp);
        }
    }
}
