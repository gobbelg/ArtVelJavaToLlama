package server;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Starts llama-server from a LlamaServerConfig, waits until the model is loaded,
 * and stops it again. Use in try-with-resources so the server is always stopped:
 *
 *   try (LlamaServerManager server = new LlamaServerManager(config, logFile)) {
 *       server.start();
 *       ... use server.baseUrl() and server.slotSize() ...
 *   }
 *
 * - The server's console output is appended to logFile (and the last lines are
 *   kept in memory to explain start-up failures).
 * - If something is already listening on the configured host:port, start()
 *   refuses to continue: an existing server's --ctx-size/--parallel are unknown,
 *   so the slot size could not be trusted. Stop it first.
 * - A JVM shutdown hook stops the server if the program exits without close().
 */
public class LlamaServerManager implements AutoCloseable {

    private static final long POLL_MILLIS = 1000;
    private static final int TAIL_LINES = 40;

    // llama-server prints e.g. "... new slot, n_ctx_slot = 12288" for each slot.
    private static final Pattern SLOT_CTX = Pattern.compile("n_ctx_slot\\s*=\\s*(\\d+)");

    private final LlamaServerConfig config;
    private final Path logFile;
    private final HttpClient http = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(2))
            .build();
    private final Deque<String> tail = new ArrayDeque<>();

    private volatile int reportedSlotSize = -1;
    private int effectiveSlotSize = -1;
    private Process process;
    private Thread shutdownHook;

    public LlamaServerManager(LlamaServerConfig config, Path logFile) {
        this.config = config;
        this.logFile = logFile;
    }

    public String baseUrl() {
        return config.baseUrl();
    }

    /**
     * Tokens available to one request. Calculated as --ctx-size / --parallel and
     * cross-checked against the n_ctx_slot value in the server's start-up log; if
     * the log reports a SMALLER value, that smaller value is used.
     */
    public int slotSize() {
        if (effectiveSlotSize < 0) {
            throw new IllegalStateException("Server not started");
        }
        return effectiveSlotSize;
    }

    public void start() throws IOException, InterruptedException {
        if (process != null) {
            throw new IllegalStateException("Server already started by this manager");
        }
        if (!Files.isRegularFile(config.executable())) {
            throw new IOException("llama-server executable not found: " + config.executable());
        }
        if (!Files.isRegularFile(config.model())) {
            throw new IOException("Model file not found: " + config.model());
        }
        if (isPortInUse()) {
            throw new IllegalStateException("Something is already listening on " + config.baseUrl()
                    + ". Stop that llama-server first: its --ctx-size/--parallel are unknown, so the "
                    + "slot size could not be trusted. (Close its console window, or end the llama-server process in "
                    + "Task Manager / Activity Monitor.)");
        }

        Path logDir = logFile.toAbsolutePath().getParent();
        if (logDir != null) {
            Files.createDirectories(logDir);
        }
        Files.writeString(logFile,
                "\n===== llama-server start " + LocalDateTime.now() + " : " + config.describe() + " =====\n"
                        + "Command: " + String.join(" ", config.commandLine()) + "\n",
                StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);

        System.out.println("Starting llama-server (" + config.describe() + ")...");
        System.out.println("  server log: " + logFile.toAbsolutePath());

        ProcessBuilder pb = new ProcessBuilder(config.commandLine());
        pb.directory(config.executable().toAbsolutePath().getParent().toFile());
        pb.redirectErrorStream(true);
        final Process p = pb.start();
        process = p;

        Thread pump = new Thread(() -> pumpOutput(p), "llama-server-log");
        pump.setDaemon(true);
        pump.start();

        shutdownHook = new Thread(this::stopProcess, "llama-server-shutdown");
        Runtime.getRuntime().addShutdownHook(shutdownHook);

        waitUntilReady();

        Thread.sleep(500); // let the log pump deliver the last start-up lines
        int computed = config.slotSize();
        int reported = reportedSlotSize;
        if (reported > 0 && reported < computed) {
            System.err.println("WARNING: the server reported n_ctx_slot = " + reported
                    + ", smaller than the calculated slot size " + computed + ". Using " + reported + ".");
            effectiveSlotSize = reported;
        } else {
            effectiveSlotSize = computed;
            if (reported < 0) {
                System.out.println("Note: no n_ctx_slot line found in the server log; "
                        + "using the calculated slot size " + computed + ".");
            } else if (reported > computed) {
                System.out.println("Note: server reported n_ctx_slot = " + reported
                        + " (shared KV cache?); keeping the conservative calculated slot size " + computed + ".");
            }
        }
        System.out.println("llama-server is ready. Slot size: " + effectiveSlotSize + " tokens.");
    }

    @Override
    public void close() {
        stopProcess();
        if (shutdownHook != null) {
            try {
                Runtime.getRuntime().removeShutdownHook(shutdownHook);
            } catch (IllegalStateException alreadyShuttingDown) {
                // JVM is exiting; the hook is running or about to run.
            }
            shutdownHook = null;
        }
    }

    // -------------------------------------------------------------------------

    private void waitUntilReady() throws IOException, InterruptedException {
        Duration startupTimeout = Duration.ofMinutes(config.startupTimeoutMinutes());
        long deadline = System.nanoTime() + startupTimeout.toNanos();
        while (System.nanoTime() < deadline) {
            if (!process.isAlive()) {
                Thread.sleep(500); // let the log pump drain
                throw new IOException("llama-server exited during start-up (exit code "
                        + process.exitValue() + "). Last log lines:\n" + tailText());
            }
            // 200 = ready; 503 = model still loading; -1 = not accepting connections yet.
            if (healthStatus() == 200) {
                return;
            }
            Thread.sleep(POLL_MILLIS);
        }
        throw new IOException("llama-server was not ready within " + startupTimeout.toMinutes()
                + " minutes. Last log lines:\n" + tailText());
    }

    private int healthStatus() throws InterruptedException {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(config.baseUrl() + "/health"))
                    .timeout(Duration.ofSeconds(5))
                    .GET()
                    .build();
            return http.send(request, HttpResponse.BodyHandlers.discarding()).statusCode();
        } catch (IOException e) {
            return -1;
        }
    }

    private boolean isPortInUse() {
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(config.host(), config.port()), 1000);
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private void pumpOutput(Process p) {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8));
                PrintWriter log = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
                        new FileOutputStream(logFile.toFile(), true), StandardCharsets.UTF_8)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                log.println(line);
                log.flush();
                synchronized (tail) {
                    tail.addLast(line);
                    if (tail.size() > TAIL_LINES) {
                        tail.removeFirst();
                    }
                }
                Matcher m = SLOT_CTX.matcher(line);
                if (m.find()) {
                    reportedSlotSize = Integer.parseInt(m.group(1));
                }
            }
        } catch (IOException e) {
            // Stream closed because the process ended; nothing more to capture.
        }
    }

    private String tailText() {
        synchronized (tail) {
            return String.join("\n", tail);
        }
    }

    private synchronized void stopProcess() {
        Process p = process;
        if (p == null || !p.isAlive()) {
            return;
        }
        p.descendants().forEach(ProcessHandle::destroy);
        p.destroy();
        try {
            if (!p.waitFor(15, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                p.waitFor(15, TimeUnit.SECONDS);
            }
        } catch (InterruptedException e) {
            p.destroyForcibly();
            Thread.currentThread().interrupt();
        }
    }
}
