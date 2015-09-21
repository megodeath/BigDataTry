package weka_bee;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.ArrayList;
import java.util.Arrays;
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

public class app_weka {

	private static final int LIM_VALUES_FOR_NOMINAL = 10000;

	private static final int TOTAL_X_DEF_64 = 63;

	public static FastVector fvWekaAttributes;

	public static void main(String[] args) throws Exception {

		List<Vector<String>> testData = parseFile(new File("train.csv").toURI().toURL(), true);

		List<Vector<String>> analyseData = parseFile(new File("test.csv").toURI().toURL(), false);

		Instances isTrainingSet = generateExampleSet(testData);
		// Instances isTestSet = generateExampleSet(analyseData);

		Classifier cModel = (Classifier) new NaiveBayes();
		cModel.buildClassifier(isTrainingSet);

		Evaluation eTest = new Evaluation(isTrainingSet);
		eTest.evaluateModel(cModel, isTrainingSet);

		String strSummary = eTest.toSummaryString();
		System.out.println(strSummary);

		double[][] cmMatrix = eTest.confusionMatrix();

		for (double[] ds : cmMatrix) {
			System.out.println(Arrays.toString(ds));
		}

		for (Vector<String> vector : analyseData) {
			Instance iExample = createInstanceFromData(vector);
			if (iExample != null) {
				iExample.setDataset(isTrainingSet);
				System.out.println(Arrays.toString(cModel.distributionForInstance(iExample)));
			}
		}

	}

	private static Instances generateExampleSet(List<Vector<String>> testData) {
		Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, 50000);
		// Set class index
		isTrainingSet.setClassIndex(62);

		for (Vector<String> vector : testData) {
			Instance iExample = createInstanceFromData(vector);
			if (iExample != null) {
				isTrainingSet.add(iExample);
			}
		}
		return isTrainingSet;
	}

	private static Instance createInstanceFromData(Vector<String> vector) {
		Instance iExample = new SparseInstance(TOTAL_X_DEF_64);
		for (int i = 0; i < vector.size(); i++) {
			if (((Attribute) fvWekaAttributes.elementAt(i)).isNumeric()) {
				try {
					iExample.setValue((Attribute) fvWekaAttributes.elementAt(i),
							Double.parseDouble(vector.elementAt(i)));
				} catch (Exception E) {
					// iExample.setValue((Attribute)
					// fvWekaAttributes.elementAt(i), 0);
					return null;
				}
			} else {
				iExample.setValue((Attribute) fvWekaAttributes.elementAt(i), vector.elementAt(i));
			}
		}
		return iExample;
	}

	public static List<Vector<String>> parseFile(URL url, boolean forLearning) throws IOException {

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
		if (forLearning) {
			fvWekaAttributes = createWekaAttributes(rawForNominal);
		}
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

}
