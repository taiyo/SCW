package scw.algorithm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class Loader {
	
	public static ClassificationDataSet getData(File file) {
		ClassificationDataSet train = null;
		String[] label = null;
		List<List<Double>> points = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			String line;
			line = br.readLine();
			label = line.trim().split(",");
			while( (line = br.readLine()) != null ) {
				String[] tmps = line.trim().split(",");
				List<Double> point = new ArrayList<>();
				for (String s : tmps) {
					point.add(Double.parseDouble(s));
				}
				points.add(point);
			}
		} catch (FileNotFoundException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		
		List<Vec> vecList = new ArrayList<>();
		for (List<Double> point : points) {
			vecList.add(new DenseVector(point));
		}
		train = new ClassificationDataSet(vecList.get(0).length(), new CategoricalData[0], new CategoricalData(2));
		for (int i = 0; i < vecList.size(); i++) {
			if (label.equals("1")) {
				train.addDataPoint(vecList.get(i), new int[0], 1);
			} else {
				train.addDataPoint(vecList.get(i), new int[0], 0);
			}
		}
		return train;
	}
}
