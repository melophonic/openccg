package plugins;

import opennlp.ccg.ngrams.*;

import java.io.*;
import java.util.*;
import java.lang.Thread;

public class MyNgramCombo extends LinearNgramScorerCombo {
	
	public static enum BigWordsLMType {
		kenlm("models/realizer/gigaword4.5g.kenlm.bin"),
		berkeleylm("models/realizer/berkeleylm.bin");

		public final String defaultPath;

		private BigWordsLMType(String defaultPath) {
			this.defaultPath = defaultPath;
		}
	}
	

	static String wordsLM() {
		return System.getProperty("words.lm", "models/realizer/train.3bo");
	}

	static String wordsSCLM() {
		return System.getProperty("words.sc.lm", "models/realizer/train-sc.3bo");
	}

	static String stposFLM() {
		return System.getProperty("stpos.flm", "models/realizer/stp3.flm");
	}

	// map to keep track of trigram model for reuse
	static Map<Thread, NgramScorer> lmMap = new IdentityHashMap<Thread, NgramScorer>(5);

	// return big lm, while setting trigram model if using it as a stand-in
	static NgramScorer getBigLM() throws IOException {

		/** default to kenlm if big.words.lm exists **/
		BigWordsLMType bigWordsLmType = BigWordsLMType.valueOf(
				System.getProperty("big.words.lm.type", BigWordsLMType.kenlm.name()));
		String bigWordsLm = System.getProperty("big.words.lm", bigWordsLmType.defaultPath);
		
		/** otherwise if any lm type's default path exists, use that **/
		boolean bigWordsLmExists = new File(bigWordsLm).exists();
		if (!bigWordsLmExists) {
			for (BigWordsLMType lmType : BigWordsLMType.values()) {
				bigWordsLmExists = new File(lmType.defaultPath).exists();
				if (bigWordsLmExists) {
					bigWordsLm = lmType.defaultPath;
					bigWordsLmType = lmType;
					break;
				}
			}
		}
		
		if (bigWordsLmExists) {
			switch (bigWordsLmType) {
				case kenlm: return new KenNgramModel(5, bigWordsLm, false, true, true, '_', false);
				case berkeleylm: return new BerkeleyNgramModel(5, bigWordsLm, false, true, true, '_', false);
			}
		}
		NgramScorer retval = new StandardNgramModel(3, wordsLM());
		lmMap.put(Thread.currentThread(), retval);
		return retval;
	}

	// return trigram lm, reusing existing one if present
	static NgramScorer getWordsLM() throws IOException {
		NgramScorer retval = lmMap.get(Thread.currentThread());
		if (retval != null) {
			lmMap.remove(Thread.currentThread());
			return retval;
		}
		return new StandardNgramModel(3, wordsLM());
	}

	public MyNgramCombo() throws IOException {
		super(new NgramScorer[] { getBigLM(), getWordsLM(),
				new StandardNgramModel(3, wordsSCLM(), true),
				new FactoredNgramModelFamily(stposFLM()) });
	}
}
