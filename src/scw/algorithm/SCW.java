package scw.algorithm;

import static java.lang.Math.*;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.distributions.Normal;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Matrix;
import jsat.linear.Vec;

public class SCW extends BaseUpdateableClassifier {

	private static final long serialVersionUID = -2762675272890619508L;
	
	private double C = 1;
    private double eta;
    private double phi, phiSqrd, zeta, psi;
    private Mode mode;
    private Vec w;
    
    /**
     * 共分散行列
     */
    private Matrix sigmaM;
    /**
     * 共分散行列の対角成分
     */
    private Vec sigmaV;
    /**
     * Sigma * x_t を格納しておく一時的なベクトル.。
     * updateメソッドで値を返す前に0を設定し直さないといけない。
     */
    private Vec Sigma_xt;
    
    /**
     * 共分散行列の対角成分だけを計算しても境界面を計算できるようだ。<br>
     * 計算量がそれにより<i>O(d<sup>2</sup>)</i>から<i>O(s)</i>となる。
     */
    private boolean diagonalOnly = false;

    /**
     * {@link #Sigma_xt} をゼロ初期化するメソッド。
     * @param x_t 
     */
    private void zeroOutSigmaXt(final Vec x_t) {
        //Zero out temp store
       if(diagonalOnly && x_t.isSparse())//only these values will be non zero 
           for(IndexValue iv : x_t)
               Sigma_xt.set(iv.getIndex(), 0.0);
       else
           Sigma_xt.zeroOut();
    }
    
    /**
     * どちらのモードのSCWを使用するのか？
     */
    public static enum Mode {
        SCWI,
        SCWII
    }

    /**
     * SCW学習器を作成する。
     * @param eta the margin confidence parameter in [0.5, 1]
     * @param mode mode controlling which algorithm to use
     * @param diagonalOnly whether or not to use only the diagonal of the 
     * covariance matrix
     * @see #setEta(double) 
     * @see #setMode(jsat.classifiers.linear.SCW.Mode) 
     * @see #setDiagonalOnly(boolean) 
     */
    public SCW(double eta, Mode mode, boolean diagonalOnly) {
        setEta(eta);
        setMode(mode);
        setDiagonalOnly(diagonalOnly);
    }
    
    /**
     * コピーコンストラクタ
     * @param other object to copy
     */
    protected SCW(SCW other) {
        this.C = other.C;
        this.diagonalOnly = other.diagonalOnly;
        this.mode = other.mode;
        this.setEta(other.eta);
        if(other.w != null)
            this.w = other.w.clone();
        if(other.sigmaM != null)
            this.sigmaM = other.sigmaM.clone();
        if(other.sigmaV != null)
            this.sigmaV = other.sigmaV.clone();
        if(other.Sigma_xt != null)
            this.Sigma_xt = other.Sigma_xt.clone();
    }

    /**
     * SCWは確率的なマージンを使い、それを最適化するように学習する。
     * なので、どのくらいのラベルが正しいかという閾値を設ける必要がある。
     * その値は、論文中のηにより設定する。<br>
     * 参考ライブラリには以下の記述がある。<br>
     * So the threshold must be in 
     * [0.5, 1.0]. Values in the range [0.8, 0.9] often work well on a wide 
     * range of problems
     * 
     * @param eta the confidence to correct to
     */
    public void setEta(double eta) {
        if(Double.isNaN(eta) || eta < 0.5 || eta > 1.0)
            throw new IllegalArgumentException("eta must be in [0.5, 1] not " + eta);
        this.eta = eta; // η
        this.phi = Normal.invcdf(eta, 0, 1); // Φ
        this.phiSqrd = phi*phi; // φ^2
        this.zeta = 1 + phiSqrd; // ζ
        this.psi  = 1 + phiSqrd/2; // ψ
    }

    /**
     * ηの値を返す。
     * @return the target correction confidence
     */
    public double getEta() {
        return eta;
    }

    /**
     * aggressiveness parameterを設定する。更新時の正則化項の役割を果す。
     * この値は＋である必要がある。
     * 
     * @param C the positive aggressiveness parameter
     */
    public void setC(double C) {
        this.C = C;
    }

    /**
     * aggressiveness parameterを返す。
     * @return the aggressiveness parameter 
     */
    public double getC() {
        return C;
    }

    /**
     * 使用するアルゴリズムの設定をする。
     * @param mode which algorithm to use
     */
    public void setMode(Mode mode) {
        this.mode = mode;
    }

    /**
     * 使用するアルゴリズムを返す。
     * @return which algorithm is used
     */
    public Mode getMode() {
        return mode;
    }

    /**
     * 共分散行列をすべて使い更新処理を行うと <i>O(d<sup>2</sup>)</i>  
     * の計算量となる（ <i>d</i>はデータの次元）。計算時間を減らすために、
     * 共分散行列の対角成分のみを使用することにより、計算量は<i>O(s)</i>となる
     * （<i>s &le; d</i> はゼロではない入力）
     * @param diagonalOnly {@code true} to use only the diagonal of the covariance
     */
    public void setDiagonalOnly(boolean diagonalOnly) {
        this.diagonalOnly = diagonalOnly;
    }

    /**
     * もし更新処理に共分散行列の対角成分のみしか使用しないまら{@code true}を返す。
     * @return {@code true} if the covariance matrix is restricted to its diagonal entries
     */
    public boolean isDiagonalOnly() {
        return diagonalOnly;
    }
    
    /**
     * 計算した重みベクトルを返す。
     * @return the learned weight vector for prediction
     */
    public Vec getWeightVec() {
        return w;
    }
    
    @Override
    public SCW clone() {
        return new SCW(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting) {
        if(numericAttributes <= 0) {
            throw new FailedToFitException("SCW requires numeric attributes to perform classification");
        } else if(predicting.getNumOfCategories() != 2) {
            throw new FailedToFitException("SCW is a binary classifier");
        }
        w = new DenseVector(numericAttributes);
        Sigma_xt = new DenseVector(numericAttributes);
        if(diagonalOnly) {
            sigmaV = new DenseVector(numericAttributes);
            sigmaV.mutableAdd(1);
        } else {
            sigmaM = Matrix.eye(numericAttributes);
        }
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        final Vec x_t = dataPoint.getNumericalValues();
        final double y_t = targetClass*2-1;
        double score = x_t.dot(w);
        
        if(diagonalOnly) {
            /* for the diagonal, its a pairwise multiplication. So just copy 
             * then multiply by the sigmas, ordes dosnt matter
             */
            if(x_t.isSparse()) {
                //Faster to set only the needed final values
                for(IndexValue iv : x_t)
                    Sigma_xt.set(iv.getIndex(), iv.getValue()*sigmaV.get(iv.getIndex()));
            } else {
                x_t.copyTo(Sigma_xt);
                Sigma_xt.mutablePairwiseMultiply(sigmaV);
            }
        } else {
            sigmaM.multiply(x_t, 1, Sigma_xt);
        }
        
        //Check for numerical issues
        double v_t = x_t.dot(Sigma_xt);
        if(v_t <= 0) {//semi positive definit, should not happen
            throw new FailedToFitException("Numerical issues occured");
        }
        
        double m_t = y_t*score;
        
        final double loss = max(0, phi*sqrt(v_t)-m_t);
        
        if(loss <= 1e-15) {
            zeroOutSigmaXt(x_t);
            return;
        }
        final double alpha_t;
        
        if(mode == Mode.SCWI) {
            double tmp = max(0, (-m_t*psi+sqrt(m_t*m_t*phiSqrd*phiSqrd/4+v_t*phiSqrd*zeta))/(v_t*zeta) );
            alpha_t = min(C, tmp);
            
        } else { //SCWII
            final double n_t = v_t+1/(2*C);
            final double gamma = phi*sqrt(phiSqrd*v_t*v_t*m_t*m_t+4*n_t*v_t*(n_t+v_t*phiSqrd));
            alpha_t = max(0, (-(2*m_t*n_t+phiSqrd*m_t*v_t)+gamma)/(2*(n_t*n_t+n_t*v_t*phiSqrd)));
        }
        
        if(alpha_t < 1e-7) {//update is numerically unstable
            zeroOutSigmaXt(x_t);
            return;
        }
        
        final double u_t = pow(-alpha_t*v_t*phi+sqrt(alpha_t*alpha_t*v_t*v_t*phiSqrd+4*v_t), 2)/4;
            
        
        
        //Now update mean and variance
        
        w.mutableAdd(alpha_t*y_t, Sigma_xt);
        
        if(diagonalOnly) { //diag does not need beta
            //Only non zeros change the cov values
            final double coef = alpha_t*phi*pow(u_t, -0.5);
            for(IndexValue iv : x_t) {
                int idx = iv.getIndex();
                double S_rr = sigmaV.get(idx);
                sigmaV.set(idx, 1/(1/S_rr+coef*pow(iv.getValue(), 2)));
            }
        } else {
            final double beta_t = alpha_t*phi/(sqrt(u_t)+v_t*alpha_t*phi);
            
            Matrix.OuterProductUpdate(sigmaM, Sigma_xt, Sigma_xt, -beta_t);
        }
        
        zeroOutSigmaXt(x_t);
    }

    @Override
    public CategoricalResults classify(DataPoint data) {
        if(w == null) {
            throw new UntrainedModelException("Model has not yet ben trained");
        }
        CategoricalResults cr = new CategoricalResults(2);
        double score = w.dot(data.getNumericalValues());
        if(score < 0) {
            cr.setProb(0, 1.0);
        } else {
            cr.setProb(1, 1.0);
        }
        return cr;
    }

    @Override
    public boolean supportsWeightedData() {
        return false;
    }
}
