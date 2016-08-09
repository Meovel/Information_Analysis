package search;

import java.io.IOException;
import java.io.StringReader;
import java.util.StringTokenizer;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.tartarus.snowball.ext.EnglishStemmer;
import org.apache.lucene.analysis.core.StopFilter;
import  org.apache.lucene.analysis.en.EnglishAnalyzer;

/**
 * @author Hao Zhang
 */

public class ZenOptimizer {

    public static String Optimize(String content) throws IOException {
        StringBuffer result = new StringBuffer();

        if (content != null && content.trim().length() > 0){
            StringReader tReader = new StringReader(content);
            EnglishAnalyzer analyzer = new EnglishAnalyzer();
            TokenStream tStream = null;
            tStream = analyzer.tokenStream("contents", tReader);
            CharTermAttribute term = tStream.addAttribute(CharTermAttribute.class);

            tStream.reset();
            while (tStream.incrementToken()){
                result.append(term.toString());
                result.append(" ");
            }
            tStream.close();
        }
        result = resubltBuffer.toString();
        if (result.length() > 0)
            result = result.substring(0, result.length()-1);

        return result;
    }

    public static void main(String[] args) {}
}