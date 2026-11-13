package config;

import java.awt.GraphicsEnvironment;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.prefs.BackingStoreException;
import java.util.prefs.Preferences;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;

/**
 * Where the {@code LLM_Trunk} folder is on this computer.
 *
 * <p>All configuration, models, llama.cpp builds, document corpora and run output live under one
 * folder that must be named {@value #FOLDER_NAME}. Its location differs from user to user and
 * computer to computer, so it is not written into the settings files. Instead the files use the
 * placeholder {@value #TOKEN} (for example {@code <LLM_TRUNK>\Run_Output}) and this class replaces
 * the placeholder with the location stored in the user's Java preferences (on Windows the
 * registry key HKEY_CURRENT_USER\Software\JavaSoft\Prefs\config).
 *
 * <p>{@link #ensure()} returns the stored location. When none is stored, or the stored folder no
 * longer exists, it opens a folder chooser. Only a folder named {@value #FOLDER_NAME} is accepted.
 * If the user cancels, a message is written to the console and the program exits.
 *
 * <p>Command-line ways to change the location:
 * <ul>
 * <li>{@code --trunk=PATH} (or {@code --trunk PATH}) on any command: validates PATH, stores it and
 * carries on with the command. See {@link #consumeArguments(String[])}.</li>
 * <li>run type {@code SET_TRUNK_LOCATION} of core.VascularRunner, or running this class's
 * {@code main} with no arguments: opens the chooser, stores the choice and exits.</li>
 * <li>run type {@code SHOW_TRUNK_LOCATION}, or {@code main} with {@code --show}: prints the stored
 * location.</li>
 * </ul>
 */
public final class TrunkLocation {

    /** The folder must have exactly this name. */
    public static final String FOLDER_NAME = "LLM_Trunk";
    /** Placeholder used in settings, batch, matrix and document-list files. */
    public static final String TOKEN = "<LLM_TRUNK>";
    /** Run types understood by core.VascularRunner. */
    public static final String RUN_SET = "SET_TRUNK_LOCATION";
    public static final String RUN_SHOW = "SHOW_TRUNK_LOCATION";

    private static final String PREF_KEY = "llmTrunkPath";
    private static final Pattern TOKEN_AND_REST =
            Pattern.compile(Pattern.quote(TOKEN) + "([^\\s\"|*?]*)", Pattern.CASE_INSENSITIVE);

    /** The location in use during this run (null until resolved). */
    private static Path root;

    private TrunkLocation() {
    }

    // -------------------------------------------------------------------------
    // Stored preference
    // -------------------------------------------------------------------------

    private static Preferences prefs() {
        return Preferences.userNodeForPackage(TrunkLocation.class);
    }

    /** True when the path is an existing folder named LLM_Trunk. */
    public static boolean isValid(Path p) {
        if (p == null) {
            return false;
        }
        Path name = p.getFileName();
        return name != null && name.toString().equals(FOLDER_NAME) && Files.isDirectory(p);
    }

    /** The stored location text, valid or not, or null when none is stored. */
    public static String storedText() {
        try {
            String s = prefs().get(PREF_KEY, null);
            return s == null || s.trim().isEmpty() ? null : s.trim();
        } catch (RuntimeException e) {
            return null;
        }
    }

    private static Path toPath(String s) {
        try {
            return s == null ? null : Paths.get(s).toAbsolutePath().normalize();
        } catch (InvalidPathException e) {
            return null;
        }
    }

    private static void store(Path p) {
        try {
            prefs().put(PREF_KEY, p.toString());
            prefs().flush();
        } catch (BackingStoreException | RuntimeException e) {
            System.err.println("WARNING: could not save the " + FOLDER_NAME + " location to the user preferences ("
                    + e.getLocalizedMessage() + "). It is used for this run only; you will be asked again next time.");
        }
    }

    // -------------------------------------------------------------------------
    // Resolving
    // -------------------------------------------------------------------------

    /**
     * The LLM_Trunk folder to use. Returns the stored location when it is valid; otherwise tells
     * the user why and opens the folder chooser. Exits the program (after a console message) if the
     * user cancels or no dialog can be shown.
     */
    public static synchronized Path ensure() {
        if (root != null && isValid(root)) {
            return root;
        }
        String text = storedText();
        Path stored = toPath(text);
        if (isValid(stored)) {
            root = stored;
            return root;
        }
        if (text == null) {
            System.out.println("The location of the " + FOLDER_NAME + " folder has not been set yet.");
        } else {
            System.out.println("The stored " + FOLDER_NAME + " location does not exist or is not named "
                    + FOLDER_NAME + ": " + text);
        }
        Path chosen = chooseOrNull(stored);
        if (chosen == null) {
            quit("No " + FOLDER_NAME + " folder was selected.");
        }
        store(chosen);
        root = chosen;
        System.out.println(FOLDER_NAME + " location set to " + root + " (saved to your user preferences).");
        return root;
    }

    /**
     * Opens the folder chooser (starting at the current location), stores the choice and returns it.
     * Returns the unchanged current location when the user cancels and one is already stored; exits
     * (with a console message) when the user cancels and none is stored.
     */
    public static synchronized Path chooseAndSave() {
        Path current = toPath(storedText());
        Path chosen = chooseOrNull(current);
        if (chosen == null) {
            if (isValid(current)) {
                root = current;
                System.out.println("Cancelled. The " + FOLDER_NAME + " location is unchanged: " + current);
                return current;
            }
            quit("No " + FOLDER_NAME + " folder was selected.");
        }
        store(chosen);
        root = chosen;
        System.out.println(FOLDER_NAME + " location set to " + root + " (saved to your user preferences).");
        return root;
    }

    /** Message on the console, then exit. */
    private static void quit(String why) {
        System.err.println(why + " The program cannot continue without it: the configuration files, models, "
                + "llama.cpp, documents and run output are all found relative to the " + FOLDER_NAME + " folder.");
        System.err.println("Run the program again and select the " + FOLDER_NAME
                + " folder, or give its location with  --trunk=<path to " + FOLDER_NAME + ">");
        System.exit(1);
    }

    /**
     * Removes {@code --trunk=PATH} / {@code --trunk PATH} from the command line. The folder must exist
     * and be named LLM_Trunk; it is stored in the user preferences and used from now on. An invalid
     * value prints a message and exits with code 2. Returns the remaining arguments. Call it first in
     * main(), together with AppSettings.consumeConfigArgument.
     */
    public static String[] consumeArguments(String[] args) {
        if (args == null) {
            return new String[0];
        }
        List<String> rest = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            String value = null;
            if (a.startsWith("--trunk=")) {
                value = a.substring("--trunk=".length());
            } else if (a.equals("--trunk") && i + 1 < args.length) {
                value = args[++i];
            }
            if (value == null) {
                rest.add(a);
                continue;
            }
            value = value.trim();
            if (value.length() >= 2 && value.startsWith("\"") && value.endsWith("\"")) {
                value = value.substring(1, value.length() - 1);
            }
            if (value.equals("~") || value.startsWith("~/") || value.startsWith("~\\")) {
                value = System.getProperty("user.home") + value.substring(1);
            }
            Path p = toPath(value);
            if (!isValid(p)) {
                System.err.println("--trunk: '" + value + "' is not an existing folder named " + FOLDER_NAME + ".");
                System.exit(2);
            }
            synchronized (TrunkLocation.class) {
                store(p);
                root = p;
            }
            System.out.println(FOLDER_NAME + " location set to " + p + " (saved to your user preferences).");
        }
        return rest.toArray(new String[0]);
    }

    /** Prints the stored location and whether it is usable. */
    public static void show() {
        String text = storedText();
        if (text == null) {
            System.out.println(FOLDER_NAME + " location: not set");
        } else {
            System.out.println(FOLDER_NAME + " location: " + text
                    + (isValid(toPath(text)) ? "" : "   (NOT FOUND, you will be asked to choose it at the next start)"));
        }
    }

    /**
     * Stand-alone use. No argument: choose and save. {@code --show}: print. {@code --trunk=PATH}: save.
     */
    public static void main(String[] args) {
        String[] rest = consumeArguments(args);
        for (String a : rest) {
            if (a.equals("--show")) {
                show();
                return;
            }
        }
        if (rest.length == args.length) {
            chooseAndSave();
        }
        System.exit(0);
    }

    // -------------------------------------------------------------------------
    // Placeholder replacement
    // -------------------------------------------------------------------------

    /**
     * Replaces {@code <LLM_TRUNK>} (any letter case) by the LLM_Trunk location and converts the path
     * characters that follow it to this computer's separator, so a file written with backslashes
     * also works on a Mac. Text without the placeholder is returned unchanged (and nothing is
     * asked). The first use may open the chooser, see {@link #ensure()}.
     */
    public static String expand(String s) {
        if (s == null || s.indexOf('<') < 0 || !TOKEN_AND_REST.matcher(s).find()) {
            return s;
        }
        String base = ensure().toString();
        Matcher m = TOKEN_AND_REST.matcher(s);
        StringBuffer out = new StringBuffer();
        while (m.find()) {
            String rest = m.group(1).replace('\\', File.separatorChar).replace('/', File.separatorChar);
            m.appendReplacement(out, Matcher.quoteReplacement(base + rest));
        }
        m.appendTail(out);
        return out.toString();
    }

    /**
     * The reverse of {@link #expand}: replaces the LLM_Trunk location inside text that is about to
     * be written to a file by {@code <LLM_TRUNK>}, so generated files stay valid on other computers.
     * Returns the text unchanged when no location is known yet.
     */
    public static String tokenize(String text) {
        Path r = root;
        if (r == null || text == null) {
            return text;
        }
        String back = r.toString();
        String fwd = back.replace('\\', '/');
        String result = text;
        for (String form : new String[] { back, fwd }) {
            Pattern p = Pattern.compile(Pattern.quote(form) + "(?=[\\\\/\\s\"|]|$)", Pattern.CASE_INSENSITIVE);
            result = p.matcher(result).replaceAll(Matcher.quoteReplacement(TOKEN));
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // The folder chooser
    // -------------------------------------------------------------------------

    /** The chosen LLM_Trunk folder, or null when the user cancelled. Exits if no dialog is possible. */
    private static Path chooseOrNull(Path start) {
        if (GraphicsEnvironment.isHeadless()) {
            System.err.println("The " + FOLDER_NAME + " location is needed but no dialog can be shown on this computer.");
            System.err.println("Start the program once with  --trunk=<path to " + FOLDER_NAME + ">  to store it.");
            System.exit(2);
        }
        final Path[] result = new Path[1];
        Runnable show = new Runnable() {
            @Override
            public void run() {
                result[0] = showChooser(start);
            }
        };
        try {
            if (SwingUtilities.isEventDispatchThread()) {
                show.run();
            } else {
                SwingUtilities.invokeAndWait(show);
            }
        } catch (Exception e) {
            System.err.println("Could not show the folder dialog: " + e);
            System.exit(2);
        }
        return result[0];
    }

    private static Path showChooser(Path start) {
        JFrame owner = new JFrame();
        owner.setAlwaysOnTop(true);
        try {
            File dir = nearestExisting(start);
            while (true) {
                JFileChooser chooser = new JFileChooser(dir) {
                    private static final long serialVersionUID = 1L;

                    @Override
                    protected JDialog createDialog(java.awt.Component parent) {
                        JDialog d = super.createDialog(parent);
                        d.setAlwaysOnTop(true);
                        return d;
                    }
                };
                chooser.setDialogTitle("Select the " + FOLDER_NAME + " folder");
                chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                chooser.setApproveButtonText("Select");
                chooser.setApproveButtonToolTipText("Use this folder as " + FOLDER_NAME);
                if (start != null && isValid(start) && dir != null && dir.equals(start.getParent() == null ? null : start.getParent().toFile())) {
                    chooser.setSelectedFile(start.toFile());
                }
                int rc = chooser.showDialog(owner, "Select");
                if (rc != JFileChooser.APPROVE_OPTION) {
                    return null;
                }
                File f = chooser.getSelectedFile();
                Path p = f == null ? null : f.toPath().toAbsolutePath().normalize();
                if (isValid(p)) {
                    return p;
                }
                JOptionPane.showMessageDialog(owner,
                        "The folder must be named \"" + FOLDER_NAME + "\".\n"
                                + (p == null ? "" : "You selected: " + p + "\n")
                                + "Please select the folder named \"" + FOLDER_NAME + "\".",
                        "Wrong folder", JOptionPane.ERROR_MESSAGE);
                dir = p == null ? dir : (Files.isDirectory(p) ? p.toFile() : nearestExisting(p));
            }
        } finally {
            owner.dispose();
        }
    }

    /** The folder to start the chooser in: the closest existing folder at or above p, else null. */
    private static File nearestExisting(Path p) {
        Path q = p;
        while (q != null) {
            if (Files.isDirectory(q)) {
                return q.toFile();
            }
            q = q.getParent();
        }
        return null;
    }

    /** For tests: forget the resolved location of this run (the stored preference is kept). */
    static synchronized void resetForTesting() {
        root = null;
    }

    /** For tests: removes the stored preference. */
    static void clearStoredForTesting() throws IOException {
        try {
            prefs().remove(PREF_KEY);
            prefs().flush();
        } catch (BackingStoreException e) {
            throw new IOException(e);
        }
    }
}
