package scw.algorithm;

import java.util.Random;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;

public class FixedProblems
{
    private static final Vec c2l_m0 = new DenseVector(new double[]{12, 14, 25, 31, 10, 9, 1});
    private static final Vec c2l_m1 = new DenseVector(new double[]{-9, -7, -13, -6, -11, -9, -1});
    private static final NormalM c2l_c0 = new NormalM(c2l_m0, Matrix.eye(c2l_m0.length()));
    private static final NormalM c2l_c1 = new NormalM(c2l_m1, Matrix.eye(c2l_m0.length()));
    
    public static ClassificationDataSet get2ClassLinear(int dataSetSize, Random rand)
    {
        ClassificationDataSet train = new ClassificationDataSet(c2l_m0.length(), new CategoricalData[0], new CategoricalData(2));
        
        for(Vec s : c2l_c0.sample(dataSetSize, rand))
            train.addDataPoint(s, new int[0], 0);
        for(Vec s : c2l_c1.sample(dataSetSize, rand))
            train.addDataPoint(s, new int[0], 1);
        
        return train;
    }
}
