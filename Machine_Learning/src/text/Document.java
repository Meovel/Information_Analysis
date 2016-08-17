/** Associate document with its term frequencies and cluster
 *
 * @author Hao Zhang
 */

package text;

import java.util.HashMap;

public class Document {
	
	public String name;
	public int cluster = 0;
	public HashMap<String, Double> tf = new HashMap<>();

	public Document(String name) {
		this.name = name;
	}

	/** Add the document to a cluster
	 *
	 * @param cluster the cluster to be added to
     */
	public void setCluster(int cluster) {
		this.cluster = cluster;
	}

	/** Add a term frequency pair to record
	 *
	 * @param t term
	 * @param f frequency
     */
	public void addTF(String t, Double f) {
		tf.put(t, f);
	}

}
