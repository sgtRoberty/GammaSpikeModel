package gammaspike.distribution;


import java.io.PrintStream;

import beast.base.core.Citation;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.core.Input.Validate;
import beast.base.evolution.speciation.SpeciesTreeDistribution;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.parameter.RealParameter;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;
import gammaspike.tree.Stubs;



@Description("Prior distribution on a birth-death tree with unobserved speciation events (stubs)")
@Citation(value =
"Douglas, J., Bouckaert, R., Harris, S.C., Carter Jr, C.W., Wills, P.R. (2024) Evolution is coupled with branching across many granularities of life. bioRxiv 2024.09.08.611933", DOI = "https://doi.org/10.1101/2024.09.08.611933",
year = 2024, firstAuthorSurname = "Douglas")
public class BirthDeathTreePriorWithStubs extends SpeciesTreeDistribution implements StubExpectation {

	final public Input<RealParameter> lambdaInput = new Input<>("lambda", "birth rate lambda");
	final public Input<RealParameter> netDiversificationRateInput = new Input<>("netDiversificationRate", "lambda - mu", Input.Validate.XOR, lambdaInput);

	final public Input<RealParameter> r0Input = new Input<>("r0", "ratio between birth and death rate");
	final public Input<RealParameter> turnoverInput = new Input<>("turnover", "inverse of r0", Input.Validate.XOR, r0Input);

	final public Input<RealParameter> samplingProportionInput = new Input<>("samplingProportion", "sampling rate psi divided by (psi + mu)", Input.Validate.OPTIONAL);

	final public Input<RealParameter> rhoInput = new Input<>("rho", "extant sampling probability rho", Validate.REQUIRED);

	final public Input<Stubs> stubsInput = new Input<>("stubs", "the stubs of this tree", Input.Validate.OPTIONAL);


	final public Input<Boolean> ignoreTreePriorInput = new Input<>("ignoreTreePrior", "ignore tree prior (debugging)", false);
	final public Input<Boolean> ignoreStubPriorInput = new Input<>("ignoreStubPrior", "ignore tree prior (debugging)", false);



	final double HEIGHT_THRESHOLD_OF_LEAF = 1e-10;
	int expectedExtantTaxaCount;

	boolean initialising = true;


	@Override
	public boolean canHandleTipDates() {
		return samplingProportionInput.get() != null;
	}

	@Override
    public void initAndValidate() {
        super.initAndValidate();

       TreeInterface tree = treeInput.get();
        if (tree == null) {
            tree = treeIntervalsInput.get().treeInput.get();
        }

		// This distribution is conditional on the number of extant taxa n, so ensure the number does not change
        expectedExtantTaxaCount = -1;

        // Ensure valid initial state
		final int MAX_ATTEMPTS = 1000;
		double lambda = this.getLambda();
		double mu = this.getMu();
		double psi = this.getPsi();
		double rho = this.getRho();

		if (psi > 0) {
			int attemptNr;
	        for (attemptNr = 0; attemptNr < MAX_ATTEMPTS; attemptNr ++) {
	        	try {
					double logP = this.getBlueTreeLogPWithSampling((Tree)tree, lambda, mu, psi, rho);
					if (logP == Double.NEGATIVE_INFINITY || logP == Double.POSITIVE_INFINITY) throw new Exception();
				} catch (Exception e) {
					double s = Randomizer.nextFloat();
					psi = s*mu / (1-s);
					continue;
				}
	        	//Log.warning("nr " + attemptNr + " " + lambda + " " + mu + " " + psi + " p=" + p);
	        	break;
	        }

	        if (attemptNr >= MAX_ATTEMPTS) {
	        	throw new IllegalArgumentException("Cannot find a valid initial state. Try tweaking lambda, mu, psi, and rho");
	        }

	        samplingProportionInput.get().setValue(psi / (psi + mu));

		}

    }


	@Override
    public double calculateTreeLogLikelihood(final TreeInterface treeInterface) {
		double logP = 0;
		Tree tree = (Tree) treeInterface;
		double lambda = this.getLambda();
		double mu = this.getMu();
		double psi = this.getPsi();
		double rho = this.getRho();

		Stubs stubs = stubsInput.get();

		// Condition checker
		if (lambda <= mu) {
			if (initialising) Log.warning("Cannot initialise because lambda is less than mu");
			initialising = false;
			return Double.NEGATIVE_INFINITY;
		}

		// This distribution is conditional on the number of extant taxa n, so ensure the number does not change
        int currentExtantTaxaCount = 0;
		for (Node leaf : tree.getNodesAsArray()) {
			if (leaf.isLeaf() && !leaf.isDirectAncestor() && leaf.getHeight() <= HEIGHT_THRESHOLD_OF_LEAF) {
				currentExtantTaxaCount ++;
			}
		}
		if (expectedExtantTaxaCount == -1) {
			expectedExtantTaxaCount = currentExtantTaxaCount;
		} else if (currentExtantTaxaCount != this.expectedExtantTaxaCount) {
			return Double.NEGATIVE_INFINITY;
		}

		double blueLogP = 0; // Probability of observing tree T conditional on number of extant taxa n
		double redLogP = 0; // Probability of observing B stubs at ages Z

		try {

			// Blue tree density
			// Probability of observing tree T conditional on number of extant taxa n
			if (!ignoreTreePriorInput.get()) {

				// Numerical stability checker
				if (Math.exp(psi) == Double.POSITIVE_INFINITY || Math.exp(-psi) == 0) {
					return Double.NEGATIVE_INFINITY;
				}

				// New method for fast calculation when psi = 0 and rho = 1
				if ((samplingProportionInput.get() == null) && (rho == 1)) {
					blueLogP = getBlueTreeLogPWithoutSampling(tree, lambda, mu);
				}
				// Method to calculate tree likelihood when psi > 0 and rho < 1
				else {
					blueLogP = getBlueTreeLogPWithSampling(tree, lambda, mu, psi, rho);
				}

			}

			// Red tree stub density
			// Probability of observing B stubs at ages Z
			if (!ignoreStubPriorInput.get() && stubs != null) {

				// Do not use the sampling model unless there is at least 1 dated tip
				// A dated tip means non-extant sampling
				boolean thereAreDatedTips = false;
				for (Node node : tree.getNodesAsArray()) {
					if (node.isLeaf() && node.getHeight() > 0) {
						thereAreDatedTips = true;
						break;
					}
				}

				// Confirm that sampled ancestor tips do not have stubs
				for (Node node : tree.getNodesAsArray()) {
					if (node.isDirectAncestor()) {
						int stubCount = stubs.getNStubsOnBranch(node.getNr());
						if (stubCount > 0) {
							//Log.warning("XXXX " + nstubs);
							return Double.NEGATIVE_INFINITY;
						}
					}
				}

				// New method for fast calculation when psi = 0 and rho = 1
				if ((samplingProportionInput.get() == null) && (rho == 1)) {
					for (int branchNr = 0; branchNr < tree.getNodeCount(); branchNr++) {
						redLogP += getLogPForBranchWithoutSampling(branchNr, lambda, mu);
					}
				}
				// Method to calculate tree likelihood when psi > 0 and rho < 1
				else {
					for (int branchNr = 0; branchNr < tree.getNodeCount(); branchNr++) {
						redLogP += getLogPForBranchWithSampling(branchNr, lambda, mu, psi, rho);
					}
				}

			}

		} catch (Exception e) {

			//Log.warning("numerical");
			//e.printStackTrace();

			// Numerical errors
			return Double.NEGATIVE_INFINITY;
		}


		if (initialising && blueLogP == Double.NEGATIVE_INFINITY) {
			Log.warning("Cannot initialise the tree prior because blue logP is negative infinity");
		}
		else if (initialising && redLogP == Double.NEGATIVE_INFINITY) {
			Log.warning("Cannot initialise the tree prior because red logP is negative infinity");
		}

		if (logP == Double.POSITIVE_INFINITY) {
			return Double.NEGATIVE_INFINITY;
		}

		logP = blueLogP + redLogP;
		initialising = false;
		return logP;

    }


	// Equation 9 from
	// Stadler, Tanja. "Sampling-through-time in birth–death trees." Journal of theoretical biology 267.3 (2010): 396-404.
	// Tree likelihood calculation when psi > 0 and rho < 1
	public double getBlueTreeLogPWithSampling(Tree tree, double lambda, double mu, double psi, double rho) throws Exception {
		double m = 0, n = 0, k = 0;
		for (Node leaf : tree.getNodesAsArray()) {
			if (!leaf.isLeaf()) continue;
			// Sampled ancestor
			if (leaf.isDirectAncestor()) {
				k++;
			}
			// Extant leaf
			else if (!leaf.isDirectAncestor() && leaf.getHeight() <= HEIGHT_THRESHOLD_OF_LEAF) {
				n++;
			}
			// Extinct leaf
			else if (!leaf.isDirectAncestor() && leaf.getLength() > HEIGHT_THRESHOLD_OF_LEAF) {
				m++;
			}
			//else {
				//Log.warning("Dev Error 3543: " + leaf.getID() +" is not a valid leaf " + leaf.getLength() + " " + leaf.isDirectAncestor());
			//}
		}
		if (m + n + k != tree.getLeafNodeCount()) {
			//Log.warning("n=" + n + " m=" + m + " k=" + k + " != " + tree.getLeafNodeCount());
			return Double.NEGATIVE_INFINITY;
		}

		double blueLogP = 0;

		// Equation 9: p(T|n)
		// Conditioned on sampling n extant individuals
		double c1 = getC1(lambda, mu, psi);
		double c2 = getC2(lambda, mu, psi, rho);
		double x1 = tree.getRoot().getHeight();
		blueLogP += Math.log(4) + Math.log(n) + Math.log(rho) + Math.log(lambda) + Math.log(psi) * (k + m); // maybe extra lambda is typo??
		blueLogP -= Math.log(c1) + Math.log(c2+1) + Math.log(1 - c2 + (1 + c2) * Math.exp(c1 * x1)); // x=x1, typo in paper

		// Loop through internal nodes including root
		for (Node internal : tree.getNodesAsArray()) {
			if (internal.isLeaf()) continue;
			// Confirm that neither child is a sampled ancestor
			if (internal.isFake()) continue;
			double nodeHeight = internal.getHeight();
			blueLogP += Math.log(lambda) + Math.log(getP1(nodeHeight, rho, c1, c2));
		}

		// Loop through extinct leaves
		for (Node leaf : tree.getNodesAsArray()) {
			if (!leaf.isLeaf()) continue;
			if (leaf.getHeight() <= HEIGHT_THRESHOLD_OF_LEAF) continue;
			if (leaf.isDirectAncestor()) continue;
			double leafHeight = leaf.getHeight();
			blueLogP += Math.log(getP0(leafHeight, lambda, mu, psi, c1, c2)) - Math.log(getP1(leafHeight, rho, c1, c2));
		}

		return blueLogP;

	}

	// New method for fast tree likelihood calculation when psi = 0 and rho = 1
	public double getBlueTreeLogPWithoutSampling(Tree tree, double lambda, double mu) {
		// Blue tree speciation density, conditional on survival of both children
		double blueLogP = -this.getBlueIntegral();

		// Loop through all branches (nodes)
		for (int branchNr = 0; branchNr < tree.getNodeCount(); branchNr++) {
			// Skip leaf nodes (tips)
			if (tree.getNode(branchNr).isLeaf()) continue;
			double nodeHeight = tree.getNode(branchNr).getHeight();
			double logRate = this.getBlueTreeLogBirthRate(nodeHeight, lambda, mu);
			blueLogP += logRate;
		}

		return blueLogP;

	}


	private double getC1(double lambda, double mu, double psi) {
		return Math.sqrt(Math.abs(Math.pow(lambda - mu - psi, 2) + 4 * lambda * psi));
	}

	private double getC2(double lambda, double mu, double psi, double rho) {
		double c1 = getC1(lambda, mu, psi);
		return -(lambda - mu - 2 * lambda * rho - psi) / c1;
	}

	// Equation 1 from
	// Stadler, Tanja. "Sampling-through-time in birth–death trees." Journal of theoretical biology 267.3 (2010): 396-404.
	private double getP0(double height, double lambda, double mu, double psi, double c1, double c2) throws Exception {
		double a = lambda + mu + psi;
		double exp = Math.exp(-c1*height)*(1-c2);
		double b = 1+c2;
		double top = a + c1*(exp - b)/(exp + b);
		double bottom = 2*lambda;
		double result = top/bottom;
		if (Double.isNaN(result) || result <= 0) {
			//Log.warning("c1=" + c1 + " c2=" + c2 + " lambda =" + lambda + " mu=" + mu + " psi=" + psi);
			throw new Exception("Numerical error: p0 is non-positive " + top + "/" + bottom + "=" + result);
		}
		return result;
	}

	// Equation 2 from
	// Stadler, Tanja. "Sampling-through-time in birth–death trees." Journal of theoretical biology 267.3 (2010): 396-404.
	private double getP1(double height, double rho, double c1, double c2) throws Exception {
		double top = 4*rho;
		double bottom = 2*(1-c2*c2) + Math.exp(-c1*height)*(1-c2)*(1-c2) + Math.exp(c1*height)*(1+c2)*(1+c2);
		double result = top/bottom;
		if (Double.isNaN(result) || result <= 0) {
			//Log.warning("c1=" + c1 + " c2=" + c2 + " lambda =" + lambda + " mu=" + mu + " psi=" + psi);
			throw new Exception("Numerical error: p1 is non-positive " + top + "/" + bottom + "=" + result);
		}
		return result;
	}


	// Equation 7 from
	// Douglas et al. "Evolution is coupled with branching across many granularities of life" (2025)
	// The (log) probability density of observing B stubs at ages Z along a single branch branchNr
	private double getLogPForBranchWithSampling(int branchNr, double lambda, double mu, double psi, double rho) {
		double logProb = 0;
		Tree tree = getTree();
		Stubs stubs = stubsInput.get();
		double treeHeight = tree.getRoot().getHeight(); // Height of the tree root
		Node node = tree.getNode(branchNr); // The node corresponding to branch branchNr

		if (!stubs.estimateStubs()) return 0;

		// Get the origin time h0 and end time h1 of branch branchNr
		double h0 = node.isRoot() ? treeHeight : node.getParent().getHeight();
		double h1 = node.getHeight();

		// If reversible jump is off
		// The number of stubs per branch is estimated
		if (!stubs.getReversibleJump()) {
			if (node.isRoot()) return 0; // If root node, return zero
			int stubCount = stubs.getNStubsOnBranch(branchNr);
			if (h0 <= h1) {
				if (stubCount == 0) return 0; // A zero-length branch must have zero stubs
				return Double.NEGATIVE_INFINITY; // Otherwise return negative infinity (impossible)
			}

			double expectedStubCount = getExpectedStubCountForBranch(h1, h0, lambda, mu, psi, rho); // Expected number of unobserved bifurcations over branchNr, E(B)

			// Poisson(expectedStubCount) distribution
			logProb = stubCount * Math.log(expectedStubCount) - expectedStubCount;
			for (int i = 2; i <= stubCount; i ++) logProb += -Math.log(i);

			return logProb;

		}

		// If reversible jump is on
		throw new IllegalStateException("The current tree prior does not allow estimation of stub placement.");

	}

	// New method for fast likelihood calculation when psi = 0 and rho = 1
	private double getLogPForBranchWithoutSampling(int branchNr, double lambda, double mu) {
		double logProb = 0;
		Tree tree = getTree();
		Stubs stubs = stubsInput.get();
		double treeHeight = tree.getRoot().getHeight(); // Height of the tree root
		Node node = tree.getNode(branchNr); // The node corresponding to branch branchNr

		if (!stubs.estimateStubs()) return 0;

		// Get the origin time h0 and end time h1 of branch branchNr
		double h0 = node.isRoot() ? treeHeight : node.getParent().getHeight();
		double h1 = node.getHeight();

		// If reversible jump is off
		// The number of stubs per branch is estimated
		if (!stubs.getReversibleJump()) {
			if (node.isRoot()) return 0; // If root node, return zero
			int stubCount = stubs.getNStubsOnBranch(branchNr);
			if (h0 <= h1) {
				if (stubCount == 0) return 0; // A zero-length branch must have zero stubs
				return Double.NEGATIVE_INFINITY; // Otherwise return negative infinity (impossible)
			}

			double expectedStubCount = getExpectedStubCountForBranch(h1, h0, lambda, mu);

			// Poisson(expectedStubCount) distribution
			logProb = stubCount * Math.log(expectedStubCount) - expectedStubCount;
			for (int i = 2; i <= stubCount; i ++) logProb += -Math.log(i);

			return logProb;

		}

		// If reversible jump is on
		throw new IllegalStateException("The current tree prior does not allow estimation of stub placement.");

	}


	public double getLambda() {
		if (this.lambdaInput.get() == null) {
			double net = netDiversificationRateInput.get().getValue();

			// Calculate without calling getMu() or there will be a never ending loop
			if (r0Input.get() != null) {
				double r = r0Input.get().getValue();
				return net / (1 - 1/r);
			}else {
				double t = turnoverInput.get().getValue();
				return net / (1 - t);
			}


		} else {
			return this.lambdaInput.get().getValue();
		}
	}
	public double getPsi() {
		if (samplingProportionInput.get() == null) return 0.0;
		double mu = this.getMu();
		double samplingProportion = samplingProportionInput.get().getValue();
		return samplingProportion*mu / (1-samplingProportion);
	}
	public double getMu() {
		double lambda = this.getLambda();
		if (r0Input.get() != null) {
			return lambda / r0Input.get().getValue();
		} else {
			return turnoverInput.get().getValue() * lambda;
		}
	}
	public double getRho() {
		return rhoInput.get().getValue();
	}
	public Tree getTree() {
		return (Tree) treeInput.get();
	}


	// Equation 1 from
	// Stolz, Stadler & Vaughan (2024) "Integrating Transmission Dynamics and Pathogen Evolution Through a Bayesian Approach" bioRxiv
	// https://doi.org/10.1101/2024.04.15.589468
	// The expected number of unobserved bifurcations on a branch between its origin time h0 and end time h1
	// when psi > 0 and rho < 1
	private double getExpectedStubCountForBranch(double h1, double h0, double lambda, double mu, double psi, double rho) {
		double c1 = getC1(lambda, mu, psi);
		double c2 = getC2(lambda, mu, psi, rho);
		double f = ((c2 - 1) * Math.exp(-c1 * h1) - c2 - 1) / ((c2 - 1) * Math.exp(-c1 * h0) - c2 - 1);
		return (h0 - h1) * (lambda + mu + psi - c1) + 2 * Math.log(f);
	}

	// Fast calculation of the expected number of unobserved bifurcations on a branch between its origin time h0 and end time h1
	// when psi = 0 and rho = 1
	private double getExpectedStubCountForBranch(double h1, double h0, double lambda, double mu) {
		Tree tree = getTree();
		double treeHeight = tree.getRoot().getHeight();
		return getRedRateIntegral(h1, lambda, mu, treeHeight) - getRedRateIntegral(h0, lambda, mu, treeHeight);
	}
	// The following class used by getExpectedStubCountForBranch when psi = 0 and rho = 1
	private double getRedRateIntegral(double height, double lambda, double mu, double finalTime) {
		double s = finalTime - height;
		double b = Math.log(lambda * Math.exp((lambda-mu) * (finalTime - s)) - mu);
		return 2 * (b + lambda * s);
	}



	// The following four classes are used by getBlueTreeLogPWithoutSampling (when psi = 0 and rho = 1)
	private double getBlueIntegral() {
		Tree tree = getTree();
		double lambda = getLambda();
		double mu = getMu();
		double treeHeight = tree.getRoot().getHeight();

		double blueIntegral = 0;
		// Loop through the nodes
		for (Node node : tree.getNodesAsArray()) {

			// Get the origin time h0 and end time h1 of branch branchNr
			double h0 = node.isRoot() ? (node.getHeight()) : node.getParent().getHeight();
			double h1 = node.getHeight();
			double integralBranch = getBlueRateIntegral(h1, lambda, mu, treeHeight) - getBlueRateIntegral(h0, lambda, mu, treeHeight);
			blueIntegral += integralBranch;

		}

		return blueIntegral;

	}
	private double getBlueRateIntegral(double height, double lambda, double mu, double rootHeight) {
        double t = rootHeight - height;
		//Log.warning(t +  " " + finalTime + " " + lambda + " " + mu);
		double tlm = t * (lambda - mu);
		double b = Math.log(lambda * Math.exp(lambda * rootHeight) - mu * Math.exp(tlm + mu * rootHeight));
		return tlm - b;
	}
	private double getBlueTreeLogBirthRate(double timeRemaining, double lambda, double mu) {
		double logQt = this.getLogQt(timeRemaining, lambda, mu);
		return Math.log(lambda) + Math.log(1 - Math.exp(logQt));
	}
	// See Remark 3.2 from
	// Stadler, Tanja. "Sampling-through-time in birth–death trees." Journal of theoretical biology 267.3 (2010): 396-404.
	// Equivalent to p0(t) when psi = 0 and rho = 1
	private double getLogQt(double time, double lambda, double mu) {
		double exp = Math.exp((lambda - mu) * time);
		double top = mu * (exp - 1);
		double btm = lambda * exp - mu;
        return Math.log(top) - Math.log(btm);
	}



	@Override
    protected boolean requiresRecalculation() {
        return super.requiresRecalculation() ||
        		InputUtil.isDirty(lambdaInput) ||
        		InputUtil.isDirty(r0Input) ||
        		InputUtil.isDirty(netDiversificationRateInput) ||
        		InputUtil.isDirty(turnoverInput) ||
        		InputUtil.isDirty(samplingProportionInput) ||
				InputUtil.isDirty(rhoInput) ||
        		InputUtil.isDirty(stubsInput);
    }



    /**
     * Loggable interface implementation follows *
     */
    @Override
    public void init(final PrintStream out) {
       // out.print(getID() + "\t" + "g0\tg1\tg2\tg3\t"+ "stubGsum\t"+ "stubH\t");
        out.print(getID() + "\t");
    }

    @Override
    public void log(final long sample, final PrintStream out) {
    	//out.print(getCurrentLogP() + "\t" + this.getG(0) + "\t" + this.getG(1) + "\t" + this.getG(2) + "\t" + this.getG(3) + "\t" + this.getGSum()  +"\t"+ this.getH() + "\t");
        out.print(getCurrentLogP() + "\t");
    }


	/**
	 * Get the expected number of unobserved speciation events
	 */
	@Override
	public double getMeanStubNumber(double nodeHeight, double parentHeight) {

		if (parentHeight <= nodeHeight) return 0;

		double lambda = this.getLambda();
		double mu = this.getMu();
		double psi = this.getPsi();
		double rho = this.getRho();

		// Fast calculation of the expected number of unobserved bifurcations
		// when psi = 0  and rho = 1
		if (psi == 0 && rho == 1) {
			return getExpectedStubCountForBranch(nodeHeight, parentHeight, lambda, mu);
		} else {
			return getExpectedStubCountForBranch(nodeHeight, parentHeight, lambda, mu, psi, rho);
		}

	}


}


