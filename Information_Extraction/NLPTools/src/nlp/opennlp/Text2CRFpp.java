/*
 * This class converts the plain text file into the CRF++ test format
 * @author Hao Zhang
 */
package nlp.opennlp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import nlp.opennlp.POSTagger.POSTagging;

public class Text2CRFpp {

	private static final String OUTPUT_PATH = "../result.crf";
	private static final String INPUT_PATH = "../testSet.txt";

	public static void main(String[] args) throws IOException {
		
		// read from input file
		File fileI = new File(INPUT_PATH);
		BufferedReader bufferI = new BufferedReader(new FileReader(fileI));
		File fileO = new File(OUTPUT_PATH);
		BufferedWriter bufferO = new BufferedWriter(new FileWriter(fileO ));
		
		String text  = "";
		String s = bufferI.readLine();
		while (s != null) {
			text = text + s;
			s = bufferI.readLine();
		}
		bufferI.close();
		
		// tokenize and tag
		POSTagger tagger = new POSTagger();
        POSTagging taggerProcess = tagger.process(text, 1);
        System.out.print(taggerProcess);
		
        // write tagged tokens to the file in CRF++ format
		for (int si = 0; si < taggerProcess._taggings.length; si++) {
			for (int ti = 0; ti < taggerProcess._taggings[si].length; ti++) {
				for (int wi = 0; wi < taggerProcess._taggings[si][ti].length; wi++) {
					bufferO.write(taggerProcess._tokens[si][wi] + "\t");
					bufferO.write(taggerProcess._taggings[si][ti][wi] + "\n");
				}
			}
			bufferO.write("\n");
		}
		bufferO.close();
        
	}

}
