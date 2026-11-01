package core;

import config.AppSettings;
import database.VascularDbImporter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Scanner;

import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

// (VascularDbImporter is in the "database" package; imported above)

public class VascularRunner {

	//@formatter:off
	  private enum RunType {
		GET_VASCULAR_PARAMETERS,
		CONVERT_JSON_TO_ARRAY,
		IMPORT_JSON_ARRAY_TO_SQL,
		RUN_THROUGHPUT_SWEEP,   // NEW: launch the throughput harness
		CANCEL;
	  }
	//@formatter:on

	private final static JFrame mainFrame = new JFrame();

	public static void main(String[] rawArgs) {

		// Removes "--config=PATH" from the arguments and records it for AppSettings
		// (same effect as -Dvascular.config=PATH).
		final String[] args = AppSettings.consumeConfigArgument(rawArgs);

		SwingUtilities.invokeLater(new Runnable() {

			@Override
			public void run() {

				RunType runType = null;

				if (args.length > 0 && RunType.valueOf(args[0]) instanceof RunType) {

					runType = RunType.valueOf(args[0]);
				} else {

					runType = selectRunType();
				}

				switch (runType) {

				case CANCEL:
					break;

				case CONVERT_JSON_TO_ARRAY: {
					try (Scanner scanner = new Scanner(System.in)) {
						System.out.print("Enter full path to *.json file: ");
						String jsonFile = scanner.next().trim();
						Path jsonFilePath = Paths.get(jsonFile);
						MainVascular.convertToJsonArray(jsonFilePath);
					} catch (IOException e) {
						System.err.println(e.getLocalizedMessage());
						e.printStackTrace();
					}
				}
					break;

				case GET_VASCULAR_PARAMETERS: {
					try {
						MainVascular.main(null);
					} catch (Exception e) {
						System.err.println(e.getLocalizedMessage());
						e.printStackTrace();
					}
				}
					break;

				case IMPORT_JSON_ARRAY_TO_SQL: {
					try (Scanner scanner = new Scanner(System.in)) {
						System.out.print("Enter full path to *.json array file: ");
						String jsonFile = scanner.next().trim();
						Path jsonArrayPath = Paths.get(jsonFile);
						if (!Files.exists(jsonArrayPath)) {
							System.err.println("File not found: " + jsonArrayPath.toAbsolutePath());
							System.exit(1);
						}
						VascularDbImporter.importJsonArray(jsonArrayPath);
					} catch (Exception e) {
						System.err.println(e.getLocalizedMessage());
						e.printStackTrace();
					}
				}
					break;

				// NEW: run the throughput sweep. Any trailing program arguments
				// after the RunType are forwarded as the THREADS values to measure
				// (e.g. "RUN_THROUGHPUT_SWEEP 3" measures only THREADS=3). With no
				// extra arguments the harness sweeps its harness.sweep-values.
				case RUN_THROUGHPUT_SWEEP: {
					try {
						String[] forwarded = (args.length > 1)
								? java.util.Arrays.copyOfRange(args, 1, args.length)
								: new String[0];
						VascularThroughputHarness.main(forwarded);
					} catch (Exception e) {
						System.err.println(e.getLocalizedMessage());
						e.printStackTrace();
					}
				}
					break;

				default:
					break;
				}

				System.exit(0);
			}

		});

	}

	private static RunType selectRunType() {

		RunType selectedRunType = null;

		mainFrame.setAlwaysOnTop(true);
		mainFrame.toFront();

		JComboBox<RunType> comboBox = new JComboBox<>(RunType.values());
		comboBox.setSelectedItem(RunType.GET_VASCULAR_PARAMETERS);

		JPanel comboBoxPanel = new JPanel();
		comboBoxPanel.add(comboBox);

		JOptionPane optionPane = new JOptionPane(comboBoxPanel, JOptionPane.QUESTION_MESSAGE,
				JOptionPane.OK_CANCEL_OPTION);

		JDialog dialog = optionPane.createDialog(mainFrame, "Select process to run");
		dialog.setAlwaysOnTop(true);
		dialog.setVisible(true);

		Object selectedDialogValue = optionPane.getValue();

		if (selectedDialogValue != null && (int) selectedDialogValue == JOptionPane.OK_OPTION) {

			selectedRunType = (RunType) comboBox.getSelectedItem();
			System.out.println("Process selected: " + selectedRunType);
		} else {

			System.out.println("User canceled the input.");
			selectedRunType = RunType.CANCEL;
		}

		mainFrame.setAlwaysOnTop(false);

		return selectedRunType;

	}
}
