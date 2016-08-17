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

    public final static String  DATA_SRC         = "data/blog_data/";
    public final static boolean REMOVE_STOPWORDS = true;
    public final static int     NUM_TOP_WORDS    = 1500;
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
        for (int i = 0; i < k; i++) {
            int rand = ThreadLocalRandom.current().nextInt(0, docs.size());
            Document doc = docs.get(rand);
            clusters.add(new Cluster(doc.tf));
            System.out.println("* Initial Centroid " + (i+1) + ": " + doc.name);
        }
        System.out.println();

        while (reassignment()) {
            for (Cluster c: clusters) c.recomputeCentroid();
        }
    }

    /** Assign each document to its closest centroid
     *
     * @return if clusters changed after the operation
     */
    public boolean reassignment() {

        boolean clustersChanged = false;

        for (Document doc: docs) {
            int currentCluster = doc.cluster;
            double currentDistance = calDistance(doc, clusters.get(currentCluster).centroid);
            clusters.get(currentCluster).addDoc(doc, currentDistance);

            for (int cluster = 0; cluster < clusters.size(); cluster++) {
                double newDistance = calDistance(doc, clusters.get(cluster).centroid);
                // TODO: Tier breaking
                if (newDistance > currentDistance) {
                    currentDistance = newDistance;
                    doc.setCluster(cluster);
                    clusters.get(currentCluster).removeDoc(doc);
                    clusters.get(cluster).addDoc(doc, newDistance);
                    currentCluster = cluster;
                    clustersChanged = true;
                }
            }
        }
        return clustersChanged;
    }

    /** Calculate the Euclidean distance (without root)
     *
     * @param doc the document of which distance to be calculated
     * @param centroidTF term frequency of current centroid
     * @return Euclidean distance
     */
    private double calDistance(Document doc, HashMap<String, Double> centroidTF) {

        double val = 0;
        HashMap<String, Double> docTF = doc.tf;

        if (docTF.size() > centroidTF.size()) {
            for (String t : centroidTF.keySet()) {
                if (docTF.get(t) != null) {
                    val += java.lang.Math.pow(docTF.get(t) - centroidTF.get(t), 2);
                }
            }
        } else {
            for (String t : docTF.keySet()) {
                if (centroidTF.get(t) != null) {
                    val += java.lang.Math.pow(docTF.get(t) - centroidTF.get(t), 2);
                }
            }
        }
        return val;
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
                if (ugCount != null) d.addTF(key, feature.getValue());
            }

            docs.set(index, d);
            index++;
        }

//        for (int i = 0; i < index; i++) {
//            System.out.println("[Doc " + i + "]");
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
