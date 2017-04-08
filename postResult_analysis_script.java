

import java.io.File;
import java.io.FileWriter;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;

import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;

public class postResult_analysis_script {
	
	// copyFiles_for_results_Analysis_Later
	public static void postResult_analysis_script(
													String    	input_File,
													int			N_offset_lower, //offset GROUND TRUTH
													int			N_offset_upper, //offset GROUND TRUTH
													int			N_less_than_in_PREDICTED_RANKING_lower, //PREDICTED
													int			N_less_than_in_PREDICTED_RANKING_upper, //PREDICTED
													String      outFile,
													boolean 	is_Append_outFile
													){
		
		try{
			System.out.println("input_File:"+input_File);
			FileWriter writer=new FileWriter(new File(outFile), is_Append_outFile);
			// Load authID_docIDCSV
			TreeMap<Integer, String> map_eachLine_=
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line( input_File,
																						-1, 
																						-1,
																						 "load bodytext",//debug_label,
																						 false // isPrintSOP
																						 );

			TreeMap<String, Integer> map_queryGT_count=new TreeMap<String, Integer>();
			int count_lesser_offset=0;
			//example lines
//			authID!!!score!!!topicID!!!authName!!!PRED_curr_q_topic!!!fileName!!!GT_curr_q_topic!!!PRED_curr_q_topic_RESET
//			861!!!11.127758699979433!!!5!!!emily!!!india(1)!!!trad_all.txt!!!india(21)!!!india(1)
			for(int seq:map_eachLine_.keySet()){
				if(seq==1) continue; // header
				String eachLine=map_eachLine_.get(seq);
				String [] s = eachLine.split("!!!");
				String GT_curr_q_topic=s[6];
				String curr_auth_name=s[3];
//				System.out.println(GT_curr_q_topic);
				String curr_queryNpredRank= s[7];
				String curr_score=s[1];
				String curr_map_queryCount=s[8];
				 
				int curr_PREDICTED_rank=
				Integer.valueOf(curr_queryNpredRank.substring(curr_queryNpredRank.indexOf("(")+1, 
																curr_queryNpredRank.indexOf(")") ));
				
				int curr_GT_rank=Integer.valueOf(GT_curr_q_topic.substring(GT_curr_q_topic.indexOf("(")+1, 
						  					GT_curr_q_topic.indexOf(")") ));
				String curr_query_topic_from_GT=
							GT_curr_q_topic.substring(0, GT_curr_q_topic.indexOf("(") );
//				System.out.println("GT ranking:" + curr_GT_rank
//									+" curr_query_topic_from_GT:"+curr_query_topic_from_GT
//									+" curr_PREDICTED_rank:"+curr_PREDICTED_rank);
				// 
				if(curr_GT_rank>=N_offset_lower && curr_GT_rank<=N_offset_upper
						&& curr_PREDICTED_rank >= N_less_than_in_PREDICTED_RANKING_lower
						&& curr_PREDICTED_rank <= N_less_than_in_PREDICTED_RANKING_upper
						){
					count_lesser_offset++;
					System.out.println("GT ranking:" + curr_GT_rank
										+" PRED_rank:"+curr_PREDICTED_rank+" cnt_<_off:"+count_lesser_offset
										+" topic:"+curr_query_topic_from_GT
										+" name:"+curr_auth_name);
					
					writer.append("GTrank:" + curr_GT_rank
										+"!!!PREDrank:"+curr_PREDICTED_rank //+" cnt_<_off:"+count_lesser_offset
										+"!!!topic:"+curr_query_topic_from_GT
										+"!!!name:"+curr_auth_name
										+"!!!score:"+ String.valueOf(curr_score).substring(0,4)
										+"!!!"+curr_map_queryCount
										+"\n");
					writer.flush();
					//
					if(!map_queryGT_count.containsKey(curr_query_topic_from_GT)){
						map_queryGT_count.put(curr_query_topic_from_GT, 1);
					}
					else{
						int cnt=map_queryGT_count.get(curr_query_topic_from_GT)+1;
						map_queryGT_count.put(curr_query_topic_from_GT, cnt);
					}
				}
				
			}
			// 
			System.out.println("cnt<off:"+count_lesser_offset +" of total="+(map_eachLine_.size()-1)
								+" off_GT:["+N_offset_lower+","+N_offset_upper+"]"
								+" off_PRED:["+N_less_than_in_PREDICTED_RANKING_lower+","+N_less_than_in_PREDICTED_RANKING_upper+"]"
								+" map_queryGT_count:"+map_queryGT_count);
			/// 
			writer.append("cnt<off:"+count_lesser_offset +" of total="+(map_eachLine_.size()-1)
							+" off_GT:["+N_offset_lower+","+N_offset_upper+"]"
							+" off_PRED:[:"+N_less_than_in_PREDICTED_RANKING_lower+","+N_less_than_in_PREDICTED_RANKING_upper+"]"
							+" map_queryGT_count:"+map_queryGT_count +"\n");
			writer.append("inFile:"+input_File+"\n\n");
			writer.flush();
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	
	// main
	public static void main(String[] args) throws Exception {
		 
		String baseFolder="";
		String input_File=""; String out_File="";
		String [] arr_Files=new String[4];
		baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
		input_File=baseFolder+"Result_Analysis/2ndAttempt/equ3/7/7_TRAD(equ3).txt";
		out_File=baseFolder +"out_Result_Analysis.txt";
		arr_Files[0]=input_File;
//		count_less_offset:28 of total=300 off_GT:[21,50] off_PRED:[:21,50] map_queryGT_count:{boko haram=5, climate change=7, crime=2, election=3, india=6, syria=5}		
//		count_less_offset:49 of total=300 off_GT:[1,20] off_PRED:[:1,50] map_queryGT_count:{boko haram=7, climate change=10, crime=7, election=5, india=13, syria=7}
//		count_less_offset:14 of total=300 off_GT:[1,20] off_PRED:[:21,50] map_queryGT_count:{boko haram=1, climate change=3, crime=1, election=1, india=5, syria=3}
		
		input_File=baseFolder+"Result_Analysis/2ndAttempt/equ2/7/7_TRAD(equ2).txt";
		arr_Files[1]=input_File;
//		count_less_offset:19 of total=300 off_GT:[21,50] off_PRED:[:21,50] map_queryGT_count:{boko haram=1, crime=1, election=5, india=3, syria=9}
//		count_less_offset:44 of total=300 off_GT:[1,20] off_PRED:[:1,50] map_queryGT_count:{boko haram=7, climate change=3, crime=9, election=5, india=12, syria=8}
//		count_less_offset:16 of total=300 off_GT:[1,20] off_PRED:[:21,50] map_queryGT_count:{boko haram=3, climate change=2, crime=4, election=2, india=3, syria=2}
		
		input_File=baseFolder+"Result_Analysis/2ndAttempt/equ1/7/7_TRAD(equ1).txt";
		arr_Files[2]=input_File;
//		count_less_offset:23 of total=300 off_GT:[21,50] off_PRED:[:21,50] map_queryGT_count:{boko haram=1, climate change=1, crime=3, election=4, india=10, syria=4}
//		count_less_offset:48 of total=300 off_GT:[1,20] off_PRED:[:1,50] map_queryGT_count:{boko haram=9, climate change=3, crime=6, election=6, india=17, syria=7}
//		count_less_offset:14 of total=300 off_GT:[1,20] off_PRED:[:21,50] map_queryGT_count:{boko haram=3, climate change=1, election=1, india=7, syria=2}
		
		input_File=baseFolder+"Result_Analysis/2ndAttempt/SimpleLDA/7/7_TRAD(lda).txt";
		arr_Files[3]=input_File;
//		count_less_offset:28 of total=300 off_GT:[21,50] off_PRED:[:21,50] map_queryGT_count:{boko haram=6, climate change=1, election=4, india=8, syria=9}	
//		count_less_offset:54 of total=300 off_GT:[1,20] off_PRED:[:1,50] map_queryGT_count:{boko haram=8, climate change=6, crime=9, election=7, india=17, syria=7}
//		count_less_offset:21 of total=300 off_GT:[1,20] off_PRED:[:21,50] map_queryGT_count:{boko haram=3, climate change=3, crime=4, election=2, india=8, syria=1}
		
		
		int Flag=2;
		// -------------------------------------------------------------------------------------------
		//	NOTE: OUTPUT of "calc_ground_truth_compare_LDA_OR_hlda_plsi" from ground_truth.java is used as input below.
		//TAKE THE input file, get the "GT_curr_q_topic" value and PRINT how many of them are below given value N
		// 
		if(Flag==1){
			//  ( comment and uncomment )
		    postResult_analysis_script(
	    								input_File,
	    								1, //  N_offset_lower value N [10,50] //ground truth 
	    								20, //  N_offset_upper value N [10,50] //GROUND TRUTH
	    								21,  // N_less_than_in_PREDICTED_RANKING_lower // PREDICTED
	    								50,   // N_less_than_in_PREDICTED_RANKING_upper // PREDICTED
	    								out_File,
	    								false // 
	    								);
		}
		 // 
		if(Flag==2){
			int len_=arr_Files.length;
			int c=0;
			while (c<len_ ) {
				boolean is_Append=false;
				// 
				if(c>0) is_Append=true;
				//  ( comment and uncomment )
			    postResult_analysis_script(
			    							arr_Files[c],
		    								51, //  N_offset_lower value N [10,50]  // GROUND TRUTH 
		    								60, //  N_offset_upper value N [10,50] // GROUND TRUTH
		    								1,  // N_less_than_in_PREDICTED_RANKING_lower // PREDICTED
		    								50,   // N_less_than_in_PREDICTED_RANKING_upper // PREDICTED
		    								out_File, //true
		    								is_Append  // is_Append_outFile
		    								);
				c++;
			}
			
		}
		
		// -------------------------------------------------------------------------------------------
	}

}
