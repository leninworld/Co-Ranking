

import static java.util.concurrent.TimeUnit.NANOSECONDS;

import java.util.TreeMap;

public class global_preprocess {
	
	// MAIN
	public static void main(String[] args) {
	
		long t0 = System.nanoTime();
		
		// MANUAL CHANGE..
		String baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
	    String inFile= baseFolder+"mergedall-all-tokens-TRAD-POLI.txt"; //MAIN ENGLISH ONLY
	    String out_repository_file=baseFolder+"repo_url_langDetected.txt";
	    String pastRan_repository_file=baseFolder+ "repo_url_langDetected.txt";	    
	    
	    String outFile_all=inFile+"_ONLY_ENGLISH.txt"; // FOR REPOSITORY RUN, THIS may not be used. //WILL NOT CREATE NEW FRESH FILE
	    String outFile_only_int_SetOftoken=inFile+"_ONLY_ENGLISH.txt";// FOR REPOSITORY RUN, THIS may not be used.//WILL NOT CREATE NEW FRESH FILE
 
	    String debugFile=baseFolder+"debug_global.txt";
	    
	    System.out.println("inFile:"+inFile);
    
    	// STEP 1:
	  // NOTE:**** start with file that has ****only 10 required topics**** contained in it.
	  // TIME CONSUMING(run on smaller dataset- 10 interested topics)  
	  // (1) pick a token (2) is english, write to another file //COMMENT AND UNCOMMENT as required
		TreeMap<Integer,Integer> mapOutStat=cleaning_and_convert.
											t2_english_only(
															inFile,
															out_repository_file,
															pastRan_repository_file,
															outFile_all, //	 NOT FOR REPOSITORY ONLY, this FILE WILL BE CREATED->is_run_for_global_repository=FALSE
															outFile_only_int_SetOftoken, //NOT FOR REPOSITORY ONLY, this FILE WILL BE CREATED->is_run_for_global_repository=FALSE
															1, // interested_token{1=bodyText in this case}
															2, //index_containing_primary_key == URL
															true, //is_run_for_global_repository
															debugFile,
															false //isSOPprint
															);
		
		System.out.println("ended...");
		
		// STEP 2: run global repository for organization , person etc.. 


     	 System.out.println("Time Taken:" 
		         			 	+ NANOSECONDS.toSeconds(System.nanoTime() - t0) +" seconds; "
					       	   	+ (NANOSECONDS.toSeconds(System.nanoTime() - t0))/60 +" minutes" );
		

    }
}