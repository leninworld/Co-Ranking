//package p6;

import static java.util.concurrent.TimeUnit.NANOSECONDS;

import java.util.TreeMap;

import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;
import crawler.readFile_load_wrapper_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;

//calc_statistics
public class calc_statistics_average_tokens_in_a_CSV_string {
	
	// calc_no_authors_in_each_network
	public static void calc_statistics_average_tokens_in_a_CSV_string(
													   String inFile_all_token
													   ){
		int cnt_TNC=0;
		int cnt_PNC=0;
		try{
			TreeMap<Integer, Integer> map_NOTagsBucket_count=new TreeMap<Integer, Integer>();
			
			TreeMap<Integer, String> map_lineno_Line=
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(inFile_all_token, 
																		-1, -1, "debug_label", false);
			 
				double sum=0; int c=0;
				for(int lineno:map_lineno_Line.keySet()){
					c++;
					String currLine=map_lineno_Line.get(lineno);
					
//					currLine=currLine.substring(0,currLine.replace("person","").replace("location ", "").indexOf("location2"));
//					
//					currLine=currLine.substring(currLine .indexOf("location2"), currLine.length() );
					 
					
					String [] s=   currLine.split(" "); //currLine.split(",");
					
					TreeMap<String , String> map_currLine_tokens=new TreeMap<String, String>();
					///take only len>3
					int c3=0;
					while(c3< s.length){
						if(s[c3].length()>4)
							map_currLine_tokens.put(s[c3], "");
						c3++;
					}
					
//					sum+=s.length; //includes all tokens
					sum+=map_currLine_tokens.size() ; //filtered only tokens>threshold
					
					
					if(!map_NOTagsBucket_count.containsKey(s.length))
						map_NOTagsBucket_count.put(s.length , 1);
					else{
						int h =map_NOTagsBucket_count.get( s.length) +1;
						map_NOTagsBucket_count.put(s.length, h);
					}
					
				}
			 
			System.out.println("Average="+sum / (double) c +" "+sum/(double)9116+" map_NOTagsBucket_count:"+map_NOTagsBucket_count);
			
			int sum_of_all_bucket=0;
			//total number of counts under each bucket
			for(int currBucket_COUNT:map_NOTagsBucket_count.keySet()){
				sum_of_all_bucket+=map_NOTagsBucket_count.get(currBucket_COUNT);
			}
			//calculate percentage of tags in each bucket
			for(int currBucket_COUNT:map_NOTagsBucket_count.keySet()){
				System.out.println("bucket of tag size:"+currBucket_COUNT +"  % w.r.t total="+ 100*(map_NOTagsBucket_count.get(currBucket_COUNT)/ (double) sum_of_all_bucket));
			}
			
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
		
		
	}
	
	// main
	public static void main(String[] args) {
		   long t0 = System.nanoTime();
		   String baseFolder="/Users/lenin/Downloads/p6/merged/dummy.mergeall/ds8/";
		   		
		   baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/debugMALLETfiles/";
		   String inFile_all_token=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens-onlyKEYWORDS.txt";
//		   inFile_all_token=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens-token4OnlyPERSONLOCATION.txt";//acual number+loc
		   inFile_all_token=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens-token4OnlyORGANIZATION.txt";
		   inFile_all_token=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens-token4OnlyPERSON.txt";
		   
		   //
		   calc_statistics_average_tokens_in_a_CSV_string(inFile_all_token);
		   
		    
		System.out.println("Time Taken:" + NANOSECONDS.toSeconds(System.nanoTime() - t0) +" seconds; "
	   				  + (NANOSECONDS.toSeconds(System.nanoTime() - t0))/60 +" minutes");
	}
	 

}
