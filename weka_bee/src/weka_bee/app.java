package weka_bee;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.io.Resources;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class app {

	private static final int LIM_VALUES_FOR_NOMINAL = 10000;

	private static final int TOTAL_X_DEF_64 = 63;

	public static FastVector fvWekaAttributes;

	public static void main(String[] args) throws Exception {

		// Declare the feature vector

		List<Vector<String>> testData = parseFile(new File("train.csv").toURI().toURL());

		// Create an empty training set
		Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, 50000);
		// Set class index
		isTrainingSet.setClassIndex(62);

		for (Vector<String> vector : testData) {
			Instance iExample = new SparseInstance(TOTAL_X_DEF_64);
			for (int i = 0; i < vector.size(); i++) {
				if (((Attribute) fvWekaAttributes.elementAt(i)).isNumeric()) {
					iExample.setValue((Attribute) fvWekaAttributes.elementAt(i),
							Double.parseDouble(vector.elementAt(i)));
				} else {
					iExample.setValue((Attribute) fvWekaAttributes.elementAt(i), vector.elementAt(i));
				}
			}
			isTrainingSet.add(iExample);
		}

		// Create a naïve bayes classifier
		Classifier cModel = (Classifier) new NaiveBayes();
		cModel.buildClassifier(isTrainingSet);

		Evaluation eTest = new Evaluation(isTrainingSet);
		eTest.evaluateModel(cModel, isTrainingSet);

		// Print the result à la Weka explorer:
		String strSummary = eTest.toSummaryString();
		System.out.println(strSummary);

		// Get the confusion matrix
		double[][] cmMatrix = eTest.confusionMatrix();

		System.out.println(cmMatrix);

		// Specify that the instance belong to the training set
		// in order to inherit from the set description

		// iUse.setDataset(isTrainingSet);

		// Get the likelihood of each classes
		// fDistribution[0] is the probability of being “positive”
		// fDistribution[1] is the probability of being “negative”
		// double[] fDistribution = cModel.distributionForInstance(iUse);

	}

	public static List<Vector<String>> parseFile(URL url) throws IOException {

		Splitter onComma = Splitter.on(",");
		List<String> raw = Resources.readLines(url, Charsets.UTF_8);
		List<Vector<String>> data = Lists.newArrayList();

		List<HashSet<String>> rawForNominal = new ArrayList<HashSet<String>>();
		for (int i = 0; i < TOTAL_X_DEF_64; i++) {
			rawForNominal.add(new HashSet<String>());
		}
		for (String line : raw.subList(1, raw.size())) {
			Vector<String> v = new Vector<String>(TOTAL_X_DEF_64);
			int i = 0;
			Iterable<String> values = onComma.split(line);
			for (String value : Iterables.limit(values, TOTAL_X_DEF_64)) {
				rawForNominal.get(i).add(value);
				v.add(i++, value);
			}
			data.add(v);
		}

		fvWekaAttributes = createWekaAttributes(rawForNominal);

		return data;
	}

	private static FastVector createWekaAttributes(List<HashSet<String>> rawForNominal) {
		FastVector fvWekaAttributes = new FastVector(TOTAL_X_DEF_64);
		for (int i = 0; i < TOTAL_X_DEF_64; i++) {
			Attribute currAttr = null;
			if (rawForNominal.get(i).size() < LIM_VALUES_FOR_NOMINAL) {
				FastVector fvNominalVal = new FastVector();
				for (String currValue : rawForNominal.get(i)) {
					fvNominalVal.addElement(currValue);
				}
				currAttr = new Attribute(Integer.toString(i), fvNominalVal);
			} else {
				currAttr = new Attribute(Integer.toString(i));
			}
			fvWekaAttributes.addElement(currAttr);
		}
		return fvWekaAttributes;
	}

	/*
	 * public static long parseUnsignedHex(String text) { if (text.length() ==
	 * 16) { return (parseUnsignedHex(text.substring(0, 1)) << 60) |
	 * parseUnsignedHex(text.substring(1)); } return Long.parseLong(text, 16); }
	 * 
	 * public static boolean isNumeric(String str) { NumberFormat formatter =
	 * NumberFormat.getInstance(); ParsePosition pos = new ParsePosition(0);
	 * formatter.parse(str, pos); return str.length() == pos.getIndex(); }
	 */

}
