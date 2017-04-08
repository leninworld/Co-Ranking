

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Map.Entry;

import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;
import crawler.sort_given_treemap;

public class p6_evaluation_pageRank {
	// convert_bucket
	public static int convert_bucket( int i, FileWriter writerDebug, boolean isDebug ){
		int value=0;
		int multipler=7;
		try{
			if(i>=1 && i<=2)
				value=46*multipler;
			else if(i>=3 && i<=4)
				value=44*multipler;
			else if(i>=5 && i<=6)
				value=42*multipler;
			else if(i>=7 && i<=8)
				value=40*multipler;
			else if(i>=9 && i<=10)
				value=38*multipler;
			else if(i>=11 && i<=12)
				value=36*multipler;
			else if(i>=13 && i<=14)
				value=34*multipler;
			else if(i>=15 && i<=16)
				value=32*multipler;
			else if(i>=17 && i<=18)
				value=30*multipler;
			else if(i>=19 && i<=20)
				value=28*multipler;
			else if(i>=21 && i<=23)
				value=26*multipler;
			else if(i>=24 && i<=25)
				value=24*multipler;
			else if(i>=26 && i<=27)
				value=22*multipler;
			else if(i>=28 && i<=29)
				value=20*multipler;
			else if(i>=30 && i<=31)
				value=18*multipler;
			else if(i>=32 && i<=33)
				value=16*multipler;
			else if(i>=34 && i<=36)
				value=14*multipler;
			else if(i>=37 && i<=39)
				value=12*multipler;
			else if(i>=40 && i<=42)
				value=10*multipler;
			else if(i>=43 && i<=45)
				value=8*multipler;
			else if(i>=46 && i<=48)
				value=6*multipler;
			else if(i>=49 && i<=50)
				value=4*multipler;
			else if(i>=50 && i<=150)
				value=26;
			else if(i>=151 && i>=200 )
				value=24;
			else if(i>=201  && i>=250  )
				value=22;
			else if(i>=251  && i>=300  )
				value=20;
			else if(i>=301  && i>=350  )
				value=18;
			else if(i>=351 )
				value=16;
			
			if(isDebug){
				writerDebug.append("\n value:"+value+" i="+i);
				writerDebug.flush();
			}
		}
		catch(Exception e){
			e.printStackTrace();
		}
			
		return value;
	}
	//split_each_file_in_FoldeR_into_TRAN_n_POLI
	private static void split_each_file_in_FoldeR_into_TRAN_n_POLI(String baseFolder, 
																   String onlyFolder_pagerankOutputTFIDF,//	INPUT FOLDER NAME ONLY (ALREADY EXISTS) NOT FULL PATH 
																   String pageRank_Files_folder,  //INPUT FOLDER (FULL PATH already exists)
																   String pattern_for_sourceChannel_TRAD, 
																   String pattern_for_sourceChannel_POLI
																   ) {
		// TODO Auto-generated method stub
		try {
			
			String newTRADfolder_create=baseFolder+onlyFolder_pagerankOutputTFIDF+"_TRAD/";
			String newPOLIfolder_create=baseFolder+onlyFolder_pagerankOutputTFIDF+"_POLI/";
			
			File TRAD_folder=new File(newTRADfolder_create);
			File POLI_folder=new File(newPOLIfolder_create);
			if(!TRAD_folder.exists())
				TRAD_folder.mkdir();
			if(!POLI_folder.exists())
				POLI_folder.mkdir();
			
			//
			File file_folder=new File(pageRank_Files_folder);
			System.out.println("pageRank_Files_folder:"+pageRank_Files_folder);
			String [] arr_files=file_folder.list();
			System.out.println("arr_files.len:"+arr_files.length);
			int c=0;
			String line="";
			//
			while(c<arr_files.length){
				
				if(arr_files[c].indexOf("_Store")>=0) {c++;continue;}
				
				BufferedReader reader=new BufferedReader(new FileReader(new File(pageRank_Files_folder+arr_files[c])));
				String TRAD_conc_STRING="";
				String POLI_conc_STRING="";
				
				while((line=reader.readLine())!=null){
					//TRAD
					if(line.indexOf(pattern_for_sourceChannel_TRAD)>=0){
						if(TRAD_conc_STRING.length()==0){
							TRAD_conc_STRING=line;
						}
						else{
							TRAD_conc_STRING=TRAD_conc_STRING+"\n"+line;
						}
					}
					// POLI
					else if(line.indexOf(pattern_for_sourceChannel_POLI)>=0){
						if(POLI_conc_STRING.length()==0){
							POLI_conc_STRING=line;
						}
						else{
							POLI_conc_STRING=POLI_conc_STRING+"\n"+line;
						}
					}
					
				}
				System.out.println("output trad:"+TRAD_folder+arr_files[c]+".txt");
				System.out.println("output poli:"+POLI_folder+arr_files[c]+".txt");
				//create a new file in destination folder
				FileWriter writer_TRAD=new FileWriter(TRAD_folder+"/"+arr_files[c]);
				FileWriter writer_POLI=new FileWriter(POLI_folder+"/"+arr_files[c]);
				System.out.println("destination TRAD:"+TRAD_folder+"/"+arr_files[c]);
				System.out.println("destination POLI:"+POLI_folder+"/"+arr_files[c]);
				
				writer_TRAD.append(TRAD_conc_STRING); writer_TRAD.flush();
				writer_POLI.append(POLI_conc_STRING);writer_POLI.flush();
				 
				c++;
			}
			
			
			
			
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
		
	}
	
	//convert_each_author_each_doc_FREQ_to_each_auth_CUMM_FREQ
	public static void convert_each_author_each_doc_FREQ_to_each_auth_CUMM_FREQ(
																		 String baseFolder,
																		 String inFile_1_for_each_doc_word_id_AND_FREQ, //be tf-ifd (or) freq 
																		 String inFile_2_for_each_auth_id_AND_doc_id_CSV,//3 columns,First=empty
																		 String outFile,
																		 boolean isAppend_outFile){
 	   TreeMap<Integer,String> mapIS_tf_idf_each_doc=new TreeMap<Integer, String>();
 	   TreeMap<Integer,String> mapIS_each_authorID_docID_CSV=new TreeMap<Integer, String>();
 	   TreeMap<Integer, TreeMap<String,Double>> mapMAINEachLine_as_Doc_word_id_tf_idf=new TreeMap<Integer, TreeMap<String,Double>>();
 	   TreeMap<Integer, TreeMap<String,Double>> mapMERGEDEachLine_as_Auth_ID_CUMM_word_id_tf_idf=new TreeMap<Integer, TreeMap<String,Double>>();
 	  
 	   int errror_count=0;
 	   
 	   TreeMap<String, Double> map_each_line_asDoc_wordID_tfidf_pairs =new TreeMap<String, Double>();
   	   int lineNumber=0; int doc_ID=0;
   	   TreeMap<Integer,String> map_each_doc_WordTFIDF_of_inFile=new TreeMap<Integer, String>();
   	   FileWriter writerDebug=null;
 	   String line=""; int lineNo=0;
		try{
			writerDebug=new FileWriter(new File(baseFolder+"debug-tf-idf.txt"), true);
			BufferedReader br=new BufferedReader(new FileReader(new File(inFile_1_for_each_doc_word_id_AND_FREQ)));
			//////////////
			writerDebug.append(  "1:input:"+inFile_1_for_each_doc_word_id_AND_FREQ+"\n"
								+"2:input:"+inFile_2_for_each_auth_id_AND_doc_id_CSV+"\n"
								+"3:output:"+outFile+" <-- inside convert_each_author_each_doc_FREQ_to_each_auth_CUMM_FREQ \n");
			writerDebug.flush();
			
           	// each doc ID -> word_id:tf_idf  OR word_id_tf_idf
//   	        mapIS_tf_idf_each_doc=
//   	        readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
//   	        .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
//   	        													  inFile_1_for_each_doc_word_id_AND_tf_idf,
//   	        													  -1, -1
//   									            				  , " load tf-idf external"
//   									            				  , true //isPrintSOP
//   									            				  );
   	        //
   	     mapIS_each_authorID_docID_CSV=
   	        readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
   	        .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
   	        													  inFile_2_for_each_auth_id_AND_doc_id_CSV,
   	        													  -1, -1
   									            				  , " load  auth id & doc id"
   									            				  , false //isPrintSOP
   									            				  );

   	     writerDebug.append("mapIS_each_authorID_docID_CSV.size:"+mapIS_each_authorID_docID_CSV.size()+" of file->"+inFile_2_for_each_auth_id_AND_doc_id_CSV+"\n");
   	     writerDebug.flush();
   	     
   	     System.out.println("Reading convert_each_author_each_doc_FREQ_to_each_auth_CUMM_FREQ() for file->inFile_1_for_each_doc_word_id_AND_FREQ");
   	   	// Convert to map inFile_1_for_each_doc_word_id_AND_tf_idf
 		while( (line = br.readLine()) !=null){
 			lineNo++;
 			map_each_line_asDoc_wordID_tfidf_pairs= new TreeMap<String, Double>();
				//System.out.println("l:"+lineNumber++);
				if(line.length()>=1){
					
					///debug
					if(lineNo<20)
						System.out.println("lineNo:"+lineNo+"  line:"+line);
					
					String [] arr_tokens=line.split("!!!"); //first token has doc_ID
					
					if(arr_tokens.length<2){
						System.out.println("arr_tokens:"+arr_tokens.length+" line:"+line);
						writerDebug.append(" not 2 token error: "+doc_ID+"\n");
						writerDebug.flush();
						continue;
					}
					doc_ID=Integer.valueOf(arr_tokens[0]);
					String [] arr_wordID_delimitr_TFIDF= arr_tokens[1].split(" ");
					int cnt=0;
					// 
					map_each_doc_WordTFIDF_of_inFile.put(doc_ID, arr_tokens[1]);
					//
					while(cnt<arr_wordID_delimitr_TFIDF.length){
						
						String curr_wordID_delimitr_TFIDF =arr_wordID_delimitr_TFIDF[cnt]; //it has "233:1"
						String[] name_value=curr_wordID_delimitr_TFIDF.split(":");
						
						if(name_value.length<2){
							writerDebug.append(" not 2 token word:freq error: "+curr_wordID_delimitr_TFIDF+"\n");
							writerDebug.flush();
							continue;
						}
						
						map_each_line_asDoc_wordID_tfidf_pairs.put(name_value[0], new Double(name_value[1]));
						cnt++;
					}
				 
				}
				
				mapMAINEachLine_as_Doc_word_id_tf_idf.put(doc_ID, map_each_line_asDoc_wordID_tfidf_pairs);
				
				if(lineNo<10) //
					System.out.println("1..."+doc_ID+" "+mapMAINEachLine_as_Doc_word_id_tf_idf.size() );
				
			} //while line-read
   	     int auth_ID=0;
   	     TreeMap<String,Double> map_2_maps_merged=new TreeMap<String, Double>();
   	     
   	     writerDebug.append("\n outside mapMAINEachLine_as_Doc_word_id_tf_idf:"+mapMAINEachLine_as_Doc_word_id_tf_idf.size()+
   	    		 				" lineNo:"+lineNo+"\n");
	     writerDebug.flush();
	     
   	    // Reading auth_id AND doc_id CSV
 		for(int i:mapIS_each_authorID_docID_CSV.keySet()){
 			    line=mapIS_each_authorID_docID_CSV.get(i);
 			    map_each_line_asDoc_wordID_tfidf_pairs= new TreeMap<String, Double>();
;
				if(line.length()>=1){
					//System.out.println("line:"+line);
					String [] arr_tokens=line.split("!!!"); //second token has doc_ID
					if(arr_tokens.length<3){ // first column is empty
						continue;
					}
					
					auth_ID=Integer.valueOf(arr_tokens[1]);//first token has doc_ID
					String [] arr_doc_ids= arr_tokens[2].split(",");
					int cnt=0;
					
					System.out.println("lineNumber:"+(lineNumber++)+" ;arr_doc_ids.len:"+arr_doc_ids.length);
					int curr_doc_id=-1;
					//
					while(cnt<arr_doc_ids.length){
						
						try {
							curr_doc_id=Integer.valueOf(arr_doc_ids[cnt]); //it has "233:1"	
						} catch (Exception e) {
							writerDebug.append(" ERROR: error in integer conversion:line->" +line  + "<- from file: "+inFile_2_for_each_auth_id_AND_doc_id_CSV
														+"\n");
							writerDebug.flush();
							errror_count++;
							// TODO: handle exception
						}
						
						//
						if(arr_doc_ids.length==1){
							map_2_maps_merged=mapMAINEachLine_as_Doc_word_id_tf_idf.get(curr_doc_id);
						}
						else if(cnt==0){	
							map_2_maps_merged=
							merge_tf_idf_vector_OR_freq_vector_of_two_Docs
										(
										mapMAINEachLine_as_Doc_word_id_tf_idf.get(curr_doc_id),
										mapMAINEachLine_as_Doc_word_id_tf_idf.get(Integer.valueOf(arr_doc_ids[cnt+1]))
										);
						}
						else{
							map_2_maps_merged=
							merge_tf_idf_vector_OR_freq_vector_of_two_Docs
										(
										mapMAINEachLine_as_Doc_word_id_tf_idf.get(curr_doc_id),
										map_2_maps_merged
										);
						}
					    
						cnt++;
					} //while(cnt<arr_doc_ids.length){
					
				}
				
				mapMERGEDEachLine_as_Auth_ID_CUMM_word_id_tf_idf.put(auth_ID, 
																	 map_2_maps_merged);
				/////
				writerDebug.append("\n outside mapMERGEDEachLine_as_Auth_ID_CUMM_word_id_tf_idf:"+mapMERGEDEachLine_as_Auth_ID_CUMM_word_id_tf_idf+"\n");
			    writerDebug.flush();
			    
			    System.out.println("2..."+auth_ID+" "+map_2_maps_merged.size());
			} // each auth_id and its doc_id CSV
 		
 			FileWriter writer_=new FileWriter(new File(outFile));
 			//writing
 			for(int auth_id:mapMERGEDEachLine_as_Auth_ID_CUMM_word_id_tf_idf.keySet()){ 				
 				writer_.append(  auth_id+"!!!"+
 							  	 mapMERGEDEachLine_as_Auth_ID_CUMM_word_id_tf_idf.get(auth_id).toString()
 							  	.replace("{", "").replace("}", "").replace("=", ":").replace(" ", "")
 							  	.replace(",", " ")
 								+"\n");
 				writer_.flush();
 			}
 		
			
		}
		catch(Exception e){
			try {
				writerDebug.append(" ERROR convert_each_author_each_doc_FREQ_to_each_auth_CUMM_FREQ() "
								  +" file1:"+inFile_1_for_each_doc_word_id_AND_FREQ+" fil2:"+inFile_2_for_each_auth_id_AND_doc_id_CSV
								  +" msg:"+ e.getMessage()
								  + "\n" );
				writerDebug.flush();
			} catch (Exception e2) {
				// TODO: handle exception
			}
			
			e.printStackTrace();
			
		}
		System.out.println( " errror_count:"+errror_count +" <--check on debug file->"+baseFolder+"debug-tf-idf.txt "
					   +" lineno:"+lineNo);
	}
	// merge two maps : map is word_id:freq OR word_id:tf-idf
	private static TreeMap<String,Double> merge_tf_idf_vector_OR_freq_vector_of_two_Docs(
									TreeMap<String,Double> map1, TreeMap<String,Double> map2) {
		TreeMap<String,Double> mapOut=new TreeMap<String, Double>();
		// TODO Auto-generated method stub
		try{
			//
			for(String i:map1.keySet()){
				
				if(map2.containsKey(i)){
					mapOut.put(i, map1.get(i)+map2.get(i));
				}
				else{
					mapOut.put(i, map1.get(i));
				}

			}
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return mapOut;
	}
	//p6_calc_mean_avg_precision_MAP_pageRank
	public static TreeMap<String,Double> p6_calc_mean_avg_precision_MAP_for_pageRank(
																String baseFolder,
				    			 								String pageRank_Files_folder,
				    			 								String ground_truthFile, //input file 
																String output_matched_pageRank_and_GroundTruth,
																String output_pageRank_MAP,
																boolean is_append_output_pageRank_MAP,
																String  out_Results,
																boolean is_append_out_Results,
																int Top_N_max_for_calc_MAP,
																boolean is_append_for_debug_file,
																boolean is_use_original_ranking,
																boolean is_convert_using_bucket_for_rankingScore
															  ){
		TreeMap<String,Double> mapOut=new TreeMap<String, Double>();
		//double cumm_avg_precision=0.0;
		String [] arr_Files= new File(pageRank_Files_folder).list();
		TreeMap<Integer, String> map_lineNo_currLine=new TreeMap<Integer, String>();
		TreeMap<String, String> map_currLine_currLine=new TreeMap<String, String>();
		FileWriter writer=null;
		FileWriter writer_output_pageRank_MAP_for_each_curr_q_topic=null;
		FileWriter writer_debug=null;
		FileWriter writer_results=null;
		TreeMap<Integer, Integer> map_i_relI=new TreeMap<Integer, Integer>();
		TreeMap<String, Double> map_query_nDCG=new TreeMap<String, Double>();
		// https://en.wikipedia.org/wiki/Discounted_cumulative_gain
		double DCG = 0.; double iDCG = 0.;
		double DCG_n=0.;int i=0; int rel_i = 0; int rel_1_for_NUMERATOR=0;// for DCG
		double IDCG_n=0.; double new_sum_nDCG_n=0.;
		// below two works together
		int MAX_NUMBER_AUTHORS=2000; // EITHER one of below two is true
//		boolean is_use_original_ranking=true; //if false, uses "MAX_NUMBER_AUTHORS"
//		boolean is_convert_using_bucket_for_rankingScore=false; //"is_use_original_ranking" has to be TRUE if this is TRUE 
		try{
			int cnt=0;
			writer=new FileWriter(new File(output_matched_pageRank_and_GroundTruth));
			writer_output_pageRank_MAP_for_each_curr_q_topic=new FileWriter(new File(output_pageRank_MAP),is_append_output_pageRank_MAP);
			writer_debug=new FileWriter(new File(baseFolder+"debug_prec_MAP_for_pageRank.txt") ,is_append_for_debug_file );
			writer_results=new FileWriter(new File(out_Results), is_append_out_Results);
			
			writer_results.append("\n\n --------------------------------------------------\n");
			
			map_currLine_currLine=
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile( 
																								  ground_truthFile
																								, -1
																								, -1
																								, "" //outfile
																								, false // is_Append_outFile
																								, false
																								, "debug_label"
																								,-1 // token for  Primary key
																								, false//isSOPprint
																								);
	 
			writer.append("AuthID!!!score!!!authName!!!sourceFileName!!!rank(GT)!!!rank(predicted)!!!curr_q_topic\n");
			writer.flush();
			int total_number_of_currTopic_counted_from_filesSize=arr_Files.length;
			double precision_at_N=0.0;
			double avg_precision=0.0;
			double mean_precision=0.0;
			double mean_avg_precision=0.0;
			TreeMap<String,Double> map_qTopicID_Precision=new TreeMap<String, Double>();
			TreeMap<String, Double> map_qTopicID_Mean_Precision=new TreeMap<String, Double>();
			double sum_nDCG_n=0.;
			//iterate each file  // - each file for one curr_q_topic
			while(cnt<arr_Files.length){
				if(arr_Files[cnt].indexOf("_Store")>=0) {cnt++;continue;}
				 
				TreeMap<String, Integer> map_curr_q_topic_lines_in_GroundTruth_authName_rankGT=new TreeMap<String, Integer>();
				String curr_q_topic=arr_Files[cnt].replace(".txt", "").replace("_auth_id_doc_id_queryTopicRelated", "");
				String curr_authName="";String curr_q_label_AND_rank="";
				
				writer_debug.append("\n ----------------- ENTERING query: "+curr_q_topic+"----------\n");
				writer_debug.flush();
				List<Entry<Integer, Integer>> map_i_relI_SORTED_descend_INT=null;
				
				int lineNo=0;
				int    curr_authID=-1;
				// curr_ground_truth_line (ONLY matching curr_q_topic)
				for(String curr_ground_truth_line:map_currLine_currLine.keySet()){
					lineNo++;
//					if(lineNo<=10)
						
					if(curr_ground_truth_line.indexOf("!!!")==-1) {
						System.out.println("SKIPPING curr_ground_truth_line:"+curr_ground_truth_line);
						continue;} //skip header
					//
					if(
//						curr_ground_truth_line.indexOf("!!!"+curr_q_topic)>=0 && 
					    curr_ground_truth_line.indexOf("!!!")>=0 ){
//						System.out.println("INSIEE ------");
						String [] s=curr_ground_truth_line.split("!!!");
						curr_authName=s[2]; //authorName			
						curr_authID= Integer.valueOf(s[0].replace("auth_id:", ""));
						
						curr_authName=p6_MAIN_Work.p6_normalize_auth_name(curr_authName).replace("auth:from: ", "")
											 .replace("auth:from:",""); 
						
						curr_q_label_AND_rank=curr_ground_truth_line.split("!!!")[3]; //contains "curr_q_label(rank)" -> syria(1)
						
						String curr_q_label_from_GT=
									curr_q_label_AND_rank.substring(0 , curr_q_label_AND_rank.indexOf("("));
						
						curr_q_label_from_GT=p6_MAIN_Work.p6_normalize_auth_name(curr_q_label_from_GT).replace("auth:from: ", "")
												    .replace("auth:from:","");
						
						int rank_from_GT=new Integer(
											curr_q_label_AND_rank.substring( curr_q_label_AND_rank.indexOf("("),
																		     curr_q_label_AND_rank.length()).replace("(", "")
																		    .replace(")", ""));
						
//						System.out.println("^^^^^^^^^^label(GT):"+curr_q_label_from_GT+" rank(GT):"+rank_from_GT);
						map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.put(curr_authName, rank_from_GT);
					}
					else{ //debug
//						System.out.println("NOT:"+curr_ground_truth_line+"<-->"+curr_ground_truth_line.indexOf(curr_q_topic)
//											+"<--->"+curr_ground_truth_line.indexOf("!!!"));
					}
					
				}

				writer_debug.append("\n PUT map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.SIZE:"+map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.size()
										+" LOADED map_currLine_currLine.SIZE():"+map_currLine_currLine.size()
										+" FROM FILE: "+ground_truthFile);
				System.out.println("file:"+arr_Files[cnt]);
				//
				map_lineNo_currLine=
				readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
				readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line( pageRank_Files_folder+arr_Files[cnt]
																							, -1
																							, -1
																							, "debug_label22"
																							, false // isPrintSOP
																						);
				writer_debug.append("\n ----Num lines loaded: "+map_lineNo_currLine.size()+" from file:"+pageRank_Files_folder+arr_Files[cnt]);
				writer_debug.flush();
				
				TreeMap<Integer, Integer> map_GroundTruthRank_vs_predictedRank=new TreeMap<Integer, Integer>();
				//System.out.println("map:"+map_lineNo_currLine);
				//map_lineNo_currLine.remove(1);
				
				int size_of_grouth_truth_of_curr_q_topic=map_lineNo_currLine.size();
				// read reverse (as first rank is in last)
				int cnt2=map_lineNo_currLine.size();
				int rank_predicted_cnt=0;
				int positionCount=0;
				//each map_lineNo_currLine (Predicted)
				while(cnt2>0){ // read from LAST line of current file
					positionCount=0;
					if(map_lineNo_currLine.get(cnt2).indexOf("AuthID")>=0){
						System.out.println("SKIPPING cnt2:"+cnt2);
						writer_debug.append("\nSKIPPING cnt2:"+cnt2+"---"+map_lineNo_currLine.get(cnt2));
						writer_debug.flush();
						cnt2--; continue;
					}
					 
					rank_predicted_cnt++;
					
					writer_debug.append("\nmap_lineNo_currLine.get(cnt2):"+map_lineNo_currLine.get(cnt2));
					writer_debug.flush();
					
					String [] s=map_lineNo_currLine.get(cnt2).split("!!!");
					//skip header
					
					curr_authName=p6_MAIN_Work.p6_normalize_auth_name(s[2]).replace("auth:from: ", "")
										 	  .replace("auth:from:","");
					
//					System.out.println(map_lineNo_currLine.get(cnt2)+"<-------->"
//									  +curr_authName+"<-->"+curr_q_label_AND_rank
//									  +" rank(GT):"+map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.get(curr_authName)
//									  +" rank_predicted_cnt:"+rank_predicted_cnt
//									  //+" size:"+map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.size()
//									  );
					
					//SOME PREDICTED RANK AT HIGHER VALUES MAY EVEN BE AVAILABLE AT ground truth
					if(map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.get(curr_authName)==null){
						writer.append(map_lineNo_currLine.get(cnt2)+"!!!"+5000 //hard-coded for NULL->map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.get(curr_authName)
									    +"!!!"+rank_predicted_cnt+"!!!"+curr_q_topic+"\n");
					}
					else{
						writer.append(map_lineNo_currLine.get(cnt2)+"!!!"+map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.get(curr_authName)
										+"!!!"+rank_predicted_cnt+"!!!"+curr_q_topic+"\n");
					}
					writer.flush();
					//there could be authorName not available in the ground truth (we have only Top N ground truth)
					if(map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.containsKey(curr_authName)){
						map_GroundTruthRank_vs_predictedRank.put(
																 map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.get(curr_authName), 
																 rank_predicted_cnt);
					}
					
					cnt2--;
				} //END WHILE
				//System.out.println("map_curr_q_topic_lines_in_GroundTruth_authName_rankGT:"+map_curr_q_topic_lines_in_GroundTruth_authName_rankGT);
				 
//				writer_debug.append("\n map_curr_q_topic_lines_in_GroundTruth_authName_rankGT:"+map_curr_q_topic_lines_in_GroundTruth_authName_rankGT);
				writer_debug.append("\n\n map_GroundTruthRank_vs_predictedRank:"+map_GroundTruthRank_vs_predictedRank);
				writer_debug.flush();
				
				int hit_count=0; int count_miss=0;

				int counter_to_see_not_exceed_max_Top_N=0;
				precision_at_N=0.0;
				
//				TreeMap<Integer, Integer> map_GroundTruthRank_vs_predictedRank_sorted_INTEGER=
//														new TreeMap<Integer, Integer>();
				//SORT 
				TreeMap<Integer,Integer> map_GroundTruthRank_vs_predictedRank_sorted=
											sort_given_treemap.sortByValue_II(map_GroundTruthRank_vs_predictedRank );
//				Map<Integer, Integer> map_GroundTruthRank_vs_predictedRank_sorted =sort_given_treemap
//																				  .sortByValue_II_method4_ascending(map_GroundTruthRank_vs_predictedRank);
				
				
				System.out.println("1:"+map_GroundTruthRank_vs_predictedRank);
				System.out.println("2:"+map_GroundTruthRank_vs_predictedRank_sorted);
				//convert to integer
//				for(Integer gt_rank22:map_GroundTruthRank_vs_predictedRank.keySet()){
//					map_GroundTruthRank_vs_predictedRank_sorted_INTEGER.put(
//														gt_rank22,
//														map_GroundTruthRank_vs_predictedRank_sorted.get(gt_rank22)
//														);
					
//				}
				int pred_rank_4_DCG=0; int hit_counter2=0;
				//////////////////////DCG preparation	
				for(Integer gt_rank22:map_GroundTruthRank_vs_predictedRank.keySet()){
					if( 	
//							gt_rank22<=Top_N_max_for_calc_MAP &&
					 	    map_GroundTruthRank_vs_predictedRank.get(gt_rank22) <= Top_N_max_for_calc_MAP  //both condition
					 	){
						hit_counter2++;
						if(is_use_original_ranking){
							if(is_convert_using_bucket_for_rankingScore){ //BUCKET
								pred_rank_4_DCG =convert_bucket(gt_rank22, writer_debug, false) ;
							}
							else{ //ORIGINAL
								pred_rank_4_DCG = gt_rank22;
							}
						}else{
								pred_rank_4_DCG = (Top_N_max_for_calc_MAP-gt_rank22)+1;
						}
						rel_i=pred_rank_4_DCG;
						if(hit_counter2==1) {
							int transformed_new_Predict_rank=MAX_NUMBER_AUTHORS-pred_rank_4_DCG;
							if(is_use_original_ranking){
								if(is_convert_using_bucket_for_rankingScore){
//									rel_1_for_NUMERATOR= convert_bucket(pred_rank_4_DCG, writer_debug, false) ; // ORIGINAL
									rel_1_for_NUMERATOR=pred_rank_4_DCG; // ORIGINAL
//									writer_debug.append("\n here 2:"+pred_rank_4_DCG+" "+rel_1_for_NUMERATOR); writer_debug.flush();
								}
								else{
									rel_1_for_NUMERATOR=pred_rank_4_DCG; // ORIGINAL
								}
							}
							else{
								rel_1_for_NUMERATOR=transformed_new_Predict_rank;
							}
						}
							//
						 
								map_i_relI.put(hit_counter2, rel_i);
						 
						
						writer_debug.append("\nYES DCG : "+Top_N_max_for_calc_MAP+" GTrank:"+gt_rank22+
											" pred_rank:"+map_GroundTruthRank_vs_predictedRank.get(gt_rank22)+
											" map_GTRank_vs_predRank:"+map_GroundTruthRank_vs_predictedRank
											+" map_i_relI:"+map_i_relI
														+" rel_i:"+rel_i);
					}
					else{
						writer_debug.append("\nNO DCG : "+Top_N_max_for_calc_MAP+" GTrank:"+gt_rank22
											+" pred_rank:"+map_GroundTruthRank_vs_predictedRank.get(gt_rank22));
					}
				}
				writer_debug.append("\n3:loop : "+map_i_relI);
				double sum_rel_by_logI=0.;   IDCG_n=0.; int ideal_i_curr=0; double rel_1_for_DENOMINATOR=0.0;
				double sum_rel_by_logI_for_IDCG=0.;
				// ----------------- DCG NUMERATOR --------------------------------------------
				for( int i_curr:map_i_relI.keySet()){
					///////////////  numerator  /////////////////////////////////////////////////////////////////////
						if(Integer.valueOf(i_curr)>1){
//							sum_rel_by_logI+=map_i_relI.get(i_curr)
//											/ (double) LongMath.log2(Long.valueOf(i_curr) , RoundingMode.UNNECESSARY) ; //(   Math.log(i_curr)/ Math.log(2) );
							
							int transformed_new_Predict_rank=MAX_NUMBER_AUTHORS-map_i_relI.get(i_curr) ;
							
							if(is_use_original_ranking){
								if(is_convert_using_bucket_for_rankingScore){
//									int new_GTrank_bucket= convert_bucket(map_i_relI.get(i_curr), writer_debug, false) ; // ORIGINAL
									sum_rel_by_logI+=map_i_relI.get(i_curr) / ( Math.log(Double.valueOf(i_curr))/ Math.log(2) );  // ORIGINAL
								}
								else{
									sum_rel_by_logI+=map_i_relI.get(i_curr) / ( Math.log(Double.valueOf(i_curr))/ Math.log(2) );  // ORIGINAL
								}
							}
							else
								sum_rel_by_logI+=transformed_new_Predict_rank/ ( Math.log(Double.valueOf(i_curr))/ Math.log(2) );
							
							writer_debug.append("\n i_curr:"+i_curr+" gtRank:"+map_i_relI.get(i_curr)
												+" (TRAN)gtRank:"+transformed_new_Predict_rank
//												+" log:"+LongMath.log2(Double.valueOf(i_curr) , RoundingMode.)
												+" log:"+(   Math.log(Double.valueOf(i_curr))/ Math.log(2) )
												+" partial:"+map_i_relI.get(i_curr)/  (   Math.log(Double.valueOf(i_curr))/ Math.log(2) )
												+" sum:"+sum_rel_by_logI);
						}
				} // END for( String i_curr:map_i_relI.keySet()){
				DCG_n= rel_1_for_NUMERATOR + sum_rel_by_logI; // NUMERATOR
				
				writer_debug.flush();
				System.out.println("4:loop");
				
				writer_debug.append("\n 4:loop:rel_1_for_NUMERATOR:"+rel_1_for_NUMERATOR+" sum_rel_by_logI:"+sum_rel_by_logI);
				writer_debug.flush();
				
				//reverse read so that 
				ArrayList<Integer> keys = new ArrayList<Integer>(map_GroundTruthRank_vs_predictedRank_sorted.keySet());
				TreeMap<Integer,Integer> map_predictedRank_vs_GroundTruthRank_SORTED_NEW=new TreeMap<Integer, Integer>();
				TreeMap<Integer,Integer> map_GroundTruthRank_vs_predictedRank_SORTED_NEW=new TreeMap<Integer,Integer>();
				
			    for(int i2=keys.size()-1; i2>=0;i2--){
		            //System.out.println("sorted pair:"+keys.get(i2)+" "+map_GroundTruthRank_vs_predictedRank_sorted.get(keys.get(i2)));
		            map_GroundTruthRank_vs_predictedRank_SORTED_NEW.put(
		            									keys.get(i2).intValue(),
		            									map_GroundTruthRank_vs_predictedRank_sorted.get(keys.get(i2)).intValue()
		            									);
		            map_predictedRank_vs_GroundTruthRank_SORTED_NEW.put( 
		            							map_GroundTruthRank_vs_predictedRank_sorted.get(keys.get(i2)).intValue(),
		            							keys.get(i2).intValue()	
		            											);
		        }
				System.out.println("5:loop");
				//////////////////////////////// IDCG //////////////////////////
			    map_i_relI_SORTED_descend_INT= sort_given_treemap.entriesSortedByValues(map_i_relI);
				Iterator<Entry<Integer, Integer>> iterator = map_i_relI_SORTED_descend_INT.iterator();
				ideal_i_curr=0;
				while (iterator.hasNext()) {
					Entry<Integer, Integer> NOW=iterator.next();
//					System.out.println("NEXT:" + NOW.getKey() +" "+NOW.getValue());
//					map_i_relI_SORTED_descend.put(Integer.valueOf( NOW.getKey()), Integer.valueOf(NOW.getValue()));
					int curr_position= NOW.getKey(); int curr_rank=NOW.getValue();
					int transformed_new_Predict_rank=MAX_NUMBER_AUTHORS-curr_rank; 
					ideal_i_curr++;
					if(Integer.valueOf(ideal_i_curr)==1) {
						if(is_use_original_ranking){
							if(is_convert_using_bucket_for_rankingScore){
//								int new_GTrank_bucket= convert_bucket( curr_rank , writer_debug, false) ; // ORIGINAL
								rel_1_for_DENOMINATOR=curr_rank; 
							}
							else
								rel_1_for_DENOMINATOR= curr_rank;  // ORIGINAL
						}
						else
							rel_1_for_DENOMINATOR= transformed_new_Predict_rank;
					}
					if(Integer.valueOf(ideal_i_curr)>1){
//						sum_rel_by_logI_for_IDCG+=map_i_relI_SORTED_descend.get(i_curr)
//											/ (double) LongMath.log2(Integer.valueOf(ideal_i_curr) , RoundingMode.CEILING) ; //(   Math.log(i_curr)/ Math.log(2) );
						double log_denominator=( Math.log(Double.valueOf(ideal_i_curr))/ Math.log(2) );
						if(log_denominator==0){}
						else{
							if(is_use_original_ranking){
								if(is_convert_using_bucket_for_rankingScore){ //BUCKET
//									int new_GTrank_bucket= convert_bucket( curr_rank , writer_debug, false) ;  
									sum_rel_by_logI_for_IDCG+=  curr_rank/log_denominator; 
								}
								else{
									sum_rel_by_logI_for_IDCG+=  curr_rank/log_denominator; // ORIGINAL
								}
							}
							else
								sum_rel_by_logI_for_IDCG+=  transformed_new_Predict_rank/log_denominator;
						}
						
						writer_debug.append("\n 5.loop ideal_i_curr:"+ideal_i_curr+" curr_rank(GT):"+curr_rank
											+" (TRANS)curr_rank:"+transformed_new_Predict_rank
	//										+" log:"+LongMath.log2(Double.valueOf(i_curr) , RoundingMode.)
											+" log:"+(   Math.log(Double.valueOf(ideal_i_curr))/ Math.log(2) )
											+" partial:"+ curr_rank/log_denominator
											+" sum:"+sum_rel_by_logI_for_IDCG
											+" curr_position:"+curr_position
												);
						
					}
				} //while (iterator.hasNext()) {
				System.out.println("10:loop");
				IDCG_n = rel_1_for_DENOMINATOR + sum_rel_by_logI_for_IDCG;
				
				writer_debug.append("\n10:loop; IDCG_n->rel_1_for_DENOMINATOR:"+rel_1_for_DENOMINATOR+" sum_rel_by_logI_for_IDCG:"+sum_rel_by_logI_for_IDCG);
				writer_debug.flush();
				
				// final nDCG_n
				double nDCG_n=DCG_n / (double) IDCG_n;
				
				writer_debug.append("\n11:loop; nDCG_n-->"+nDCG_n); writer_debug.flush();
				
				map_query_nDCG.put(curr_q_topic, nDCG_n); //query-wise nDCG
				 
				writer_debug.append("\n ----- DCG_n:"+DCG_n+" IDCG_n:"+IDCG_n+" map_query_nDCG:"+map_query_nDCG);
				writer_debug.flush();
				
			    System.out.println("3:"+map_GroundTruthRank_vs_predictedRank_SORTED_NEW);
			    System.out.println("4:"+map_predictedRank_vs_GroundTruthRank_SORTED_NEW);
				
				//calc MAP (MEAN AVG PRECISION) for current Query_topic (curr file)
				for(int gt_pred_Rank:map_predictedRank_vs_GroundTruthRank_SORTED_NEW.keySet()){
					 
					int gt_Rank=map_predictedRank_vs_GroundTruthRank_SORTED_NEW.get(gt_pred_Rank);
					//
					if( 	gt_Rank<=Top_N_max_for_calc_MAP &&
						 	map_GroundTruthRank_vs_predictedRank.get(gt_Rank) <= Top_N_max_for_calc_MAP ){
					
//					System.out.println("gt_Rank_str:"+gt_Rank_str);
//					int gt_Rank=new Integer(gt_Rank_str);
					counter_to_see_not_exceed_max_Top_N++;
					//
					if(counter_to_see_not_exceed_max_Top_N>Top_N_max_for_calc_MAP){
						System.out.println(" exceeded given Top_N_max_for_calc_MAP>Top_N_max_for_calc_MAP ->"
										  +counter_to_see_not_exceed_max_Top_N+" "+Top_N_max_for_calc_MAP);
						break;
					}
					
					//System.out.println(gt_Rank+" "+map_GroundTruthRank_vs_predictedRank.get(gt_Rank));
					writer_output_pageRank_MAP_for_each_curr_q_topic.append(
																	"\n GT vs predict:"
																	+gt_Rank+" "+map_GroundTruthRank_vs_predictedRank_SORTED_NEW.get(gt_Rank) 
																	+" for curr_q_topic:"+curr_q_topic);
					writer_output_pageRank_MAP_for_each_curr_q_topic.flush();
					
//					if(size_of_grouth_truth_of_curr_q_topic<Top_N_max_for_calc_MAP)
//						Top_N_max_for_calc_MAP=size_of_grouth_truth_of_curr_q_topic;
					
					//
					if(gt_Rank<=Top_N_max_for_calc_MAP 
					  && map_GroundTruthRank_vs_predictedRank_SORTED_NEW.get(gt_Rank)<=Top_N_max_for_calc_MAP ){
						
						int pred_Rank=map_GroundTruthRank_vs_predictedRank_SORTED_NEW.get(gt_Rank);
						positionCount++;
						hit_count++;
						
						precision_at_N= hit_count/new Double(Top_N_max_for_calc_MAP);
						mean_precision=mean_precision+(hit_count/ new Double( pred_Rank ));//positionCount
						
						System.out.println("HIT: GT:"+gt_Rank+" pred:"+map_GroundTruthRank_vs_predictedRank_SORTED_NEW.get(gt_Rank)
											+" hit_count:"+hit_count+" precision_at_N:"+precision_at_N
											+" mean_precision:"+mean_precision);
						writer_debug.append("\n HIT: GT:"+gt_Rank+" pred:"+map_GroundTruthRank_vs_predictedRank_SORTED_NEW.get(gt_Rank)
											+" hit_count:"+hit_count+" precision_at_N:"+precision_at_N
											+" mean_precision:"+mean_precision);
						writer_debug.flush();
						
					}

					}
				} //curr_q_topic
				
				sum_nDCG_n=0.;
				//remove infinity to zero
				for(String cquery:map_query_nDCG.keySet()){
					String value= String.valueOf(map_query_nDCG.get(cquery));
//					System.out.println("value:"+value);
					if(value.toLowerCase().indexOf("infinit")>=0 || value.toLowerCase().indexOf("nan")>=0)
						map_query_nDCG.put(cquery, 0.);
				}
				
				//DCG (final)
				if(is_use_original_ranking==true){ 
					for(String cQuery:map_query_nDCG.keySet()){
						sum_nDCG_n+=map_query_nDCG.get(cQuery);
					}
					new_sum_nDCG_n=sum_nDCG_n;
				}
				else{ //TRANSFORMTION
					
					for(String cQuery:map_query_nDCG.keySet()){
						sum_nDCG_n+=map_query_nDCG.get(cQuery);
					}
					// TRANSFORMATION (NORMALIZE EACH VALUE
					for(String cQuery:map_query_nDCG.keySet()){
						double curr_value=0.0;
						if(sum_nDCG_n>0)
							curr_value= map_query_nDCG.get(cQuery) / (double)sum_nDCG_n;
						
						map_query_nDCG.put(cQuery, curr_value); //reinsert NORMALIZED VALUE
					}
					//new
//					for(String cQuery:map_query_nDCG.keySet()){
//						new_sum_nDCG_n+=map_query_nDCG.get(cQuery);
//					}
					new_sum_nDCG_n=sum_nDCG_n;
				}
				
				map_i_relI=new TreeMap<Integer, Integer>(); //reset
				//averaging
				if(hit_count>0)
					mean_precision=mean_precision/new Double(hit_count);
				else
					mean_precision=0.;
				
	
				
				System.out.println(" FINAL precision_at_N:"+precision_at_N
									+" mean_precision:"+mean_precision
									);
				System.out.println("# (AVG) nDCG_n:"+new_sum_nDCG_n / (double) map_query_nDCG.size() + " size:"+map_query_nDCG.size()
									+ " map_query_nDCG:"+map_query_nDCG);

				writer_results.append("\n # (AVG) nDCG_n:"+new_sum_nDCG_n / (double) map_query_nDCG.size() + " size:"+map_query_nDCG.size()
								+ " map_query_nDCG:"+map_query_nDCG);
				writer_results.flush();
				
				map_qTopicID_Precision.put(curr_q_topic, precision_at_N);//overwrite
				map_qTopicID_Mean_Precision.put(curr_q_topic, mean_precision);
				
				mean_precision=0;
				//calc prec
//				precision=hit_count/new Double(map_GroundTruthRank_vs_predictedRank.size());
//				avg_precision=avg_precision+precision;
				// mean average prec
			
				System.out.println("\navg_precision:"+avg_precision +" for query:"+curr_q_topic 
								+"\ncount_hit:"+hit_count+" Top_N_max_for_calc_MAP:"+Top_N_max_for_calc_MAP
								+" size_of_grouth_truth_of_curr_q_topic:"+size_of_grouth_truth_of_curr_q_topic
								+" map_GroundTruthRank_vs_predictedRank.size:"+map_GroundTruthRank_vs_predictedRank.size()
								+"\nmean_precision:"+mean_precision
								+" Precision@N; N="+positionCount
								+" mean_avg_precision:"+mean_avg_precision
								+" *precision_at_N:"+precision_at_N
								+" *mean_precision:"+mean_precision
								);
				
				writer_debug.append("\navg_precision:"+avg_precision +" for query:"+curr_q_topic 
									+"\ncount_hit:"+hit_count+" Top_N_max_for_calc_MAP:"+Top_N_max_for_calc_MAP
									+" size_of_grouth_truth_of_curr_q_topic:"+size_of_grouth_truth_of_curr_q_topic
									+" map_GroundTruthRank_vs_predictedRank.size:"+map_GroundTruthRank_vs_predictedRank.size()
									+"\nmean_precision:"+mean_precision
									+" Precision@N; N="+positionCount
									+" mean_avg_precision:"+mean_avg_precision
									+" *precision_at_N:"+precision_at_N
									+" *mean_precision:"+mean_precision
								  );
				writer_debug.flush();

				System.out.println("map_GroundTruthRank_vs_predictedRank:"+map_GroundTruthRank_vs_predictedRank);
				System.out.println("map_qTopicID_Precision:"+map_qTopicID_Precision);
				System.out.println("----------------------------"+(map_GroundTruthRank_vs_predictedRank.size()));
				
				cnt++;
			}//while(cnt<arr_Files.length){ - each file for one curr_q_topic
 
			System.out.println("------------------------------------");
			mean_avg_precision=0.0;
			avg_precision=0.;
			for(String curr_q_topic_1:map_qTopicID_Precision.keySet()){
				System.out.println(curr_q_topic_1+" <-> Prec:"+map_qTopicID_Precision.get(curr_q_topic_1)
									+" meanPrec:"+map_qTopicID_Mean_Precision.get(curr_q_topic_1));
				
				writer_results.append("\n"+curr_q_topic_1+" <-> Prec:"+map_qTopicID_Precision.get(curr_q_topic_1)
										+" meanPrec:"+map_qTopicID_Mean_Precision.get(curr_q_topic_1));
				
				avg_precision=avg_precision+map_qTopicID_Precision.get(curr_q_topic_1);
				
				if(!map_qTopicID_Mean_Precision.get(curr_q_topic_1).toString().equalsIgnoreCase("nan"))
					mean_avg_precision=mean_avg_precision+map_qTopicID_Mean_Precision.get(curr_q_topic_1);
			}
			
			System.out.println(
								"** avg_precision:"+avg_precision/new Double(map_qTopicID_Precision.size())
								+" of size="+map_qTopicID_Precision.size()
								+" mean_avg_precision(MAP):"+mean_avg_precision/new Double(map_qTopicID_Mean_Precision.size())
								+" of size="+map_qTopicID_Precision.size()
								);
			writer_results.append("\n ** avg_precision:"+avg_precision/new Double(map_qTopicID_Precision.size())
								+" of size="+map_qTopicID_Precision.size()
								+" mean_avg_precision(MAP):"+mean_avg_precision/new Double(map_qTopicID_Mean_Precision.size())
								+" of size="+map_qTopicID_Precision.size()
								);
			
			writer_results.append("\n # (AVG) nDCG_n:"+new_sum_nDCG_n / (double) map_query_nDCG.size() + " size:"+map_query_nDCG.size()
											+ " map_query_nDCG:"+map_query_nDCG);
			writer_results.flush();
			
			//OUT
			mapOut.put("precision", avg_precision/new Double(map_qTopicID_Precision.size()));
			mapOut.put("map", mean_avg_precision/new Double(map_qTopicID_Mean_Precision.size()));
			
			int cnt10=0;
//			while(cnt10<arr_Files.length){
//				System.out.println("file:"+arr_Files[cnt10]);
//				cnt10++;
//			}
			
			writer_results.flush();
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return mapOut;
	}

	// MAIN
    public static void main(String args[]) throws Exception {
    	 String baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
    	 String inFile_1_for_each_doc_word_id_AND_tf_idf=baseFolder+"doc_id_word_id_AND_freq.txt";
    	 String inFile_2_for_each_auth_id_AND_doc_id_CSV=baseFolder+"auth_id_doc_id.txt"; ////3 columns,First=empty
    	 
    	 inFile_2_for_each_auth_id_AND_doc_id_CSV=baseFolder+"australia_CSV.txt";
    	 
    	 int Flag=2; // {1,2}
    	 int Top_N_max_for_calc_MAP=50;
    	 String outFile=baseFolder+"tf-idf-for-australia.txt";
    	 boolean isAppend_outFile=false;
    	 //CUMM TFIDF -- convert_each_author_each_doc_tfIDF_to_each_auth_CUMM_tfIDF
//    	 convert_each_author_each_doc_tfIDF_to_each_auth_CUMM_tfIDF(
//					   baseFolder,
//					   inFile_1_for_each_doc_word_id_AND_tf_idf, //be tf-ifd (or) freq 
//					   inFile_2_for_each_auth_id_AND_doc_id_CSV,
//					   outFile,
//					   isAppend_outFile
//					   );
    	 
    	//this folder already exists before this script ran
    	 String onlyFolder_pagerankOutputTFIDF="pagerankOutputTFIDF";
     	//this folder already exists before this script ran
    	 String pageRank_Files_folder=baseFolder+onlyFolder_pagerankOutputTFIDF+"/";//each file named curr_q_topic (example:"syria") has pageRank result for it.
    	 // manually change START-------------------------------------------------------
    	 int GT_num=8;
    	 boolean is_use_original_ranking=true; //if false, uses "MAX_NUMBER_AUTHORS"
 		 boolean is_convert_using_bucket_for_rankingScore=false; //"is_use_original_ranking" has to be TRUE if this is TRUE
    	 String groundTruthFolder="ground_truth_run1";
    	 //below one obsolete
    	 String ground_truthFile=baseFolder+groundTruthFolder+"/Ground_Truth_final_"+GT_num+"_OUT.txt_Top_N.txt";//this uses POLI and TRAD together(obsolete)
    	 String ground_truthFile_TRAD=baseFolder+groundTruthFolder+"/Ground_Truth_final_"+GT_num+"_OUT.txt_trad_.txt_Top_N.txt";
    	 String ground_truthFile_POLI=baseFolder+groundTruthFolder+"/Ground_Truth_final_"+GT_num+"_OUT.txt_poli_.txt_Top_N.txt";
    	 
    	 // manually change END-------------------------------------------------------
    	 
    	 String output_matched_pageRank_and_GroundTruth_FOLDER="pagerank.output.meanAveragePrecision";
    	 if(!new File(baseFolder+output_matched_pageRank_and_GroundTruth_FOLDER).exists()) 
    		 new File(baseFolder+output_matched_pageRank_and_GroundTruth_FOLDER).mkdir(); //CREATE FOLDER
    	 String output_matched_pageRank_and_GroundTruth_TRAD=baseFolder+output_matched_pageRank_and_GroundTruth_FOLDER+"/pagerank_rankGT_rankPredicted_TRAD.txt";
    	 String output_matched_pageRank_and_GroundTruth_POLI=baseFolder+output_matched_pageRank_and_GroundTruth_FOLDER+"/pagerank_rankGT_rankPredicted_POLI.txt";
    	 String output_pageRank_MAP=baseFolder+output_matched_pageRank_and_GroundTruth_FOLDER+"/pageRank.MAP.txt";
		 String pattern_for_sourceChannel_TRAD="trad_";
         String pattern_for_sourceChannel_POLI="poli_";
         //
    	 if(Flag==1 || Flag==99){
	    	 //STEP 1 : split each files in this folder to the 
	    	 split_each_file_in_FoldeR_into_TRAN_n_POLI(
	    			 									baseFolder,
	    			 									onlyFolder_pagerankOutputTFIDF,
	    			 									pageRank_Files_folder,
	    			 									pattern_for_sourceChannel_TRAD,
	    			 									pattern_for_sourceChannel_POLI
	    			 									);
    	 }
    	 // ****DONT CHANGE BELOW, this same is used in inside --- split_each_file_in_FoldeR_into_TRAN_n_POLI()
			String newfolder_create_TRAD=baseFolder+onlyFolder_pagerankOutputTFIDF+"_TRAD/";
			String newfolder_create_POLI=baseFolder+onlyFolder_pagerankOutputTFIDF+"_POLI/";
//			String[] arr_folders=new String[2];
//			arr_folders[0]=newTRADfolder_create;
//			arr_folders[1]=newPOLIfolder_create;
			int c=0;
			
			//STEP 2 : mean AVERAGE precision for TRAD and POLI
//			// process TRAD and POLI
//			while(c< arr_folders.length){
//				String curr_pageRank_Files_folder=arr_folders[c];
//				
			String out_Results=baseFolder+output_matched_pageRank_and_GroundTruth_FOLDER+"/results.txt";
			
			if(Flag==2 || Flag==99){
					//TRAD (comment/uncomment)
				   TreeMap<String, Double> mapOUT_TRAD=
				    p6_calc_mean_avg_precision_MAP_for_pageRank( 
					    			 									baseFolder,
					    			 									newfolder_create_TRAD, //IN FOLDER
					    			 									ground_truthFile_TRAD, //INPUT
					    			 									output_matched_pageRank_and_GroundTruth_TRAD, //OUT
					    			 									output_pageRank_MAP, // OUT
					    			 									false, // output_pageRank_MAP
					    			 									out_Results,
					    			 									false,  //is_append_out_Results
					    			 									Top_N_max_for_calc_MAP,
					    			 									false,   // is_append_for_debug_file
					    			 									is_use_original_ranking,
					    			 									is_convert_using_bucket_for_rankingScore
																);
				   double prec_1=mapOUT_TRAD.get("precision"); double map_1=mapOUT_TRAD.get("map");
				    
					//POLI (comment/uncomment)
				   TreeMap<String, Double> mapOUT_POLI=
				    p6_calc_mean_avg_precision_MAP_for_pageRank(
					    			 									baseFolder,
					    			 									newfolder_create_POLI, //IN FOLDER
					    			 									ground_truthFile_POLI, //INPUT
					    			 									output_matched_pageRank_and_GroundTruth_POLI, //OUT
					    			 									output_pageRank_MAP, // OUT 
					    			 									true, // output_pageRank_MAP
					    			 									out_Results,
					    			 									true,  //is_append_out_Results
					    			 									Top_N_max_for_calc_MAP,
					    			 									true,   // is_append_for_debug_file
					    			 									is_use_original_ranking,
					    			 									is_convert_using_bucket_for_rankingScore
																);
				   double prec_2=mapOUT_POLI.get("precision"); 
				   double map_2=mapOUT_POLI.get("map");
				   
				   if(String.valueOf(prec_1) ==null) prec_1=0.0;if(String.valueOf(prec_2) ==null) prec_2=0.0;
				   
				   System.out.println("---------------------------------------------------------------------");
				   System.out.println("prec_1(TRAD):"+prec_1+" prec_2(POLI):"+prec_2+" precision(avg):"+ (prec_1+prec_2)/2.0  );
				   System.out.println("map_1(TRAD):"+map_1+" map_2(POLI):"+map_2+" precision(avg):"+ (map_1+map_2)/2.0  );
				   
				    
			}
			
			System.out.println("--->Results stored in "+out_Results+" Top_N_max_for_calc_MAP:"+Top_N_max_for_calc_MAP);
				    
		    	 //p6_calc_mean_avg_precision_MAP_for_pageRank (BELOW RUNS for together TRAD and POLI, hence commented)
		//    	 p6_calc_mean_avg_precision_MAP_for_pageRank( 
		//	    			 									baseFolder,
		//	    			 									pageRank_Files_folder, //IN FOLDER
		//	    			 									ground_truthFile, //INPUT
		//	    			 									output_matched_pageRank_and_GroundTruth, //OUT
		//	    			 									output_pageRank_MAP, // OUT 
		//	    			 									Top_N_max_for_calc_MAP
		//												    );
				
		 
//			}
    		
			System.out.println(" You ran for Top_N_max_for_calc_MAP="+Top_N_max_for_calc_MAP);
    	 }

    	 
    	
}

