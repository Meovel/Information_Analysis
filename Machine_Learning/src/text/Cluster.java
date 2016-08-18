/** Cluster and methods associated with it
 *
 * @author Hao Zhang
 */

package text;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class Cluster {
	
	public HashMap<String, Double> centroid = new HashMap<>();
	HashMap<Document, Double> docs = new HashMap<> ();
	
	public Cluster(HashMap<String, Double> c){
		centroid = c;
	}

	/** Size of the cluster
	 *
	 * @return number of docs in the cluster
     */
	public int size() {
		return docs.size();
	}

	/** Add a document and its distance from the centroid to the cluster
	 *
	 * @param doc document to be added to the cluster
	 * @param dis distance from the centroid
     */
	public void addDoc(Document doc, double dis) {
		docs.put(doc, dis);
	}

	/** Remove a document from the cluster
	 *
	 * @param d document to be removed from the cluster
     */
	public void removeDoc(Document d) {
		docs.remove(d);		
	}

	/** Recompute the centroid of the cluster
	 *  as the average of vectors
	 */
	public void recomputeCentroid() {
		HashMap<String, Double> newCentroid = new HashMap<>();

		for (Document d: docs.keySet()) {
			HashMap<String, Double> tf = d.tf;

			for (String t: tf.keySet()) {
				if (newCentroid.get(t) == null) {
					newCentroid.put(t, tf.get(t));
				} else {
					newCentroid.put(t, newCentroid.get(t) + tf.get(t));
				}
			}
		}

		int size = docs.size();
		for (String s : newCentroid.keySet()) {
			newCentroid.put(s, newCentroid.get(s) / size);
		}
		centroid = newCentroid;
	}

	/** Output top document names in the cluster
	 *
	 * @param m number of maximum outputs in each cluster
	 */
	public void outputCluster(int m) {
		// TODO: sort hashmap by value
		List mapValues = new ArrayList(docs.values());
		Collections.sort(mapValues);
		Collections.reverse(mapValues);
		
		int n = mapValues.size();
		if (m > n) m = n;

		for (int i = 0; i < m; i++){
			for (Document d : docs.keySet()) {
				if (mapValues.get(i) == docs.get(d)) {
					System.out.print("<" + d.name + ">  ");
					System.out.println("(" + mapValues.get(i) + ") ");
					break;
				}
			}
		}
	}

}
