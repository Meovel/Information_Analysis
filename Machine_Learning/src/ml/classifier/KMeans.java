/** K-means clustering
 *
 * @author Hao Zhang
 */

package ml.classifier;

import text.*;
import util.DocUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

public class KMeans {

    public final static String  DATA_SRC         = "data/blog_data_test/";
    public final static boolean REMOVE_STOPWORDS = true;
    public final static int     NUM_TOP_WORDS    = 250;
    public final static int     NUM_MAX_TF       = 10;
    public final static int     NUM_MAX_REASS    = 30;
    public final static int     NUM_CLUSTERS_K   = 3;
    public final static int     NUM_OUTPUT_DOC   = 5;

    public ArrayList<Cluster> clusters = new ArrayList<>();
    public ArrayList<Document> docs = new ArrayList<>();


    public static void main(String[] args) {

        KMeans classifier = new KMeans();
        classifier.classify(NUM_CLUSTERS_K);

        classifier.outputClusters();
    }


    /** Repeat K-means classification until no further change
     *  initial centroids are randomly chosen
     *
     * @param k number of clusters
     */
    public void classify(int k) {

        generateFeatureVectors();

        System.out.println();
        int seg = docs.size()/k;
        for (int i = 0; i < k; i++) {
            int rand = ThreadLocalRandom.current().nextInt(seg*i, seg*(i+1));
            Document doc = docs.get(rand);
            clusters.add(new Cluster(doc.tf));
            System.out.println("* Initial Centroid " + (i+1) + ": " + doc.name);
        }
        System.out.println();

        int i = 0;
        System.out.print("Iteration ");

        while (i<NUM_MAX_REASS) {
            if (!reAssignment()) {
                System.out.print("Converges ");
                break;
            }

            for (Cluster c: clusters) c.recomputeCentroid();

            i++;
            System.out.print(i + " - ");
        }

        System.out.println("Done");
        System.out.println();
    }

    /** Assign each document to its closest centroid
     *
     * @return if clusters changed after the operation
     */
    public boolean reAssignment() {

        boolean clustersChanged = false;

        for (Document doc: docs) {
            int currentClusterNo = doc.cluster;
            double currentDistance = cosDistance(doc, clusters.get(currentClusterNo).centroid);
            double originalDistance = currentDistance;
            double newDistance = currentDistance;

            for (int cluster = 0; cluster < clusters.size(); cluster++) {
                if (cluster == doc.cluster) continue;

                newDistance = cosDistance(doc, clusters.get(cluster).centroid);
                // TODO: Tier breaking
                if (newDistance != 0 && newDistance > currentDistance) {
                    currentDistance = newDistance;
                    currentClusterNo = cluster;
                }
            }

            boolean docClusterChanged = (originalDistance != currentDistance);

            if (docClusterChanged && clusters.get(doc.cluster).size()>3) {
                clusters.get(currentClusterNo).addDoc(doc, newDistance);
                clusters.get(doc.cluster).removeDoc(doc);
                doc.setCluster(currentClusterNo);
                clustersChanged = true;
            } else
                clusters.get(doc.cluster).addDoc(doc, currentDistance);
        }
        return clustersChanged;
    }

    /** Calculate the Cosine similarity (un-square-rooted)
     *
     * @param doc the document of which distance to be calculated
     * @param centroidTF term frequency of current centroid
     * @return Cosine similarity, 0 as infinity
     */
    private double cosDistance(Document doc, HashMap<String, Double> centroidTF) {

        double dotProd = 0;
        double docMod = 0;
        double centroidMod = 0;
        HashMap<String, Double> docTF = doc.tf;
        HashMap<String, Double> smallerTF;

        if (docTF.size() > centroidTF.size()) smallerTF = centroidTF;
        else smallerTF = docTF;

        for (String t: smallerTF.keySet()) {
            if (docTF.get(t) != null && centroidTF.get(t) != null)
                dotProd += docTF.get(t)*centroidTF.get(t);
        }
        dotProd = Math.pow(dotProd, 2);

        for (String t : centroidTF.keySet()) {
            centroidMod += Math.pow(centroidTF.get(t), 2);
        }
//        for (String t : docTF.keySet()) {
//            docMod += Math.pow(docTF.get(t), 2);
//        }

        return dotProd/(centroidMod*doc.mod);
    }

    /** Get feature vectors and store using sparse data structure
     *
     */
    public void generateFeatureVectors() {

        UnigramBuilder unigram = new UnigramBuilder(DATA_SRC, NUM_TOP_WORDS, REMOVE_STOPWORDS);
        int index = 0;

        for (File f: unigram.files) {
            Document d = new Document(f.getName());
            docs.add(d);

            String file_content = DocUtils.ReadFile(f);
            Map<Object, Double> featureMap = DocUtils.ConvertToFeatureMap(file_content);

            for (Map.Entry<Object, Double> feature: featureMap.entrySet()) {
                String key = (String) feature.getKey();
                Integer ugCount = unigram._topWord2Index.get(key);
                if (ugCount != null) {
                    double featureValue = feature.getValue();
                    if (NUM_MAX_TF > 0 && featureValue > NUM_MAX_TF) featureValue = NUM_MAX_TF;
                    d.addTF(key, featureValue);
                }
            }

            docs.set(index, d);
            d.calMod();
            index++;
        }

//        for (int i = 0; i < index; i++) {
//            System.out.println("<" + docs.get(i).name + ">");
//            System.out.println(docs.get(i).tf);
//        }
    }

    public void outputClusters() {

        for (int i = 0; i < clusters.size(); i++) {
            System.out.println("--- Cluster " + (i+1) + " ---");
            clusters.get(i).outputCluster(NUM_OUTPUT_DOC);
            System.out.println();
        }
    }

}
