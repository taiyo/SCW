package scw.algorithm;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Random;

import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;

import org.junit.Test;

public class SCWTest {

	@Test
	public void test() {
		
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());
     
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

        for (SCW.Mode mode : SCW.Mode.values())
        {   
            SCW scwFull = new SCW(0.9, mode, false);
            scwFull.trainC(train);
            
            for (DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), scwFull.classify(dpp.getDataPoint()).mostLikely());
        }
	}
	
	@Test
	public void testLoadingData() {
		
		ClassificationDataSet train = Loader.getData(new File("test/data/digits_train.csv"));
	     
        ClassificationDataSet test = Loader.getData(new File("test/data/digits_test.csv"));

        for (SCW.Mode mode : SCW.Mode.values())
        {   
          SCW scwFull = new SCW(0.9, mode, false);
          scwFull.trainC(train);
            
          for (DataPointPair<Integer> dpp : test.getAsDPPList())
        	  assertEquals(dpp.getPair().longValue(), scwFull.classify(dpp.getDataPoint()).mostLikely());
        }
	}

}
