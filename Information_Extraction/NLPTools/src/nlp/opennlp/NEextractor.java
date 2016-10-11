/*
 * This class extracts the result predicted by CRF++ pipeline
 * in category of Location, Organization, Person, MISC
 * refer to NE-ExtractorFormat as example output
 * @author Hao Zhang
 */

package nlp.opennlp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

public class NEextractor {

	private static final String OUTPUT_PATH = "../NE-Extractor-Result.txt";
	private static final String INPUT_PATH = "../tagged.txt";

	static HashMap<String, Integer> loc = new HashMap<String, Integer>();
	static HashMap<String, Integer> org = new HashMap<String, Integer>();
	static HashMap<String, Integer> per = new HashMap<String, Integer>();
	static HashMap<String, Integer> misc = new HashMap<String, Integer>();

	static String last_word = "";
	static String last_tag = "";
	static String last_type = "";
	
	public static void main(String[] args) throws IOException {
		
		File fileI = new File(INPUT_PATH);
		BufferedReader bufferR = new BufferedReader(new FileReader(fileI));
		File fileO = new File(OUTPUT_PATH);
		BufferedWriter bufferW = new BufferedWriter(new FileWriter(fileO));

		// read from CRF++ predicated result
		String s = bufferR.readLine();
		while (s != null) {
			process(s);
			s = bufferR.readLine();
		}
		bufferR.close();
		
		System.out.println(loc);
		System.out.println(org);
		System.out.println(per);
		System.out.println(misc);

		outputResult("LOCATION", loc, bufferW);
		outputResult("ORGANIZATION", org, bufferW);
		outputResult("PERSON", per, bufferW);
		outputResult("MISC", misc, bufferW);
		
		bufferW.close();
	}

	// process the results of crf_test to hash maps of different entity categories
	private static void process(String s) {
		StringTokenizer st = new StringTokenizer(s);
		int i = 0;
		String word = "";
		String full_tag = "";
		String tag = "";

		while (st.hasMoreTokens()) {
			i++;
			if (i == 1) {
				word = st.nextToken();
			} else if (i == 3) {
				full_tag = st.nextToken();
			} else {
				st.nextToken();
			}
		}
		//System.out.println(word + "\t" + tag);

		int index = full_tag.indexOf("-");
		if (index != -1) {
			tag = full_tag.substring(index+1, full_tag.length());
			if (last_type.equals("B") && tag.equals(last_tag)) {
				last_word += " " + word;
			} else {
				if (last_word != "" && !tag.equals(last_tag)) {
					classifyWord(last_word, last_tag);
				}

				classifyWord(word, tag);
				last_word = word;
				last_tag = tag;
				last_type = full_tag.substring(0, index);
			}
		}
	}

	private static void classifyWord(String word, String tag) {
		if (tag.equals("ORG")) {
			addToMap(word, org);
		} else if (tag.equals("LOC")) {
			addToMap(word, loc);
		} else if (tag.equals("PER")) {
			addToMap(word, per);
		} else if (tag.equals("MISC")) {
			addToMap(word, misc);
		}
	}

	private static void addToMap(String word, HashMap<String, Integer> map) {
		if (map.containsKey(word)) {
			map.put(word, map.get(word)+1);
		} else {
			map.put(word, 1);
		}
	}

	// output the result in the required format to a file
	private static void outputResult(String s, HashMap<String, Integer> hm, BufferedWriter bw) throws IOException {
		bw.write(s + "\n");
		for (Map.Entry<String, Integer> entry : hm.entrySet()) {
			String key = entry.getKey();
			String value = entry.getValue().toString();
			bw.write("\t" + key + "\t" + value + "\n");
		}
		bw.write("\n");
	}

}
