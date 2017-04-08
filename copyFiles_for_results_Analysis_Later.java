

import java.io.File;

import org.apache.commons.io.FileUtils;

public class copyFiles_for_results_Analysis_Later {
	
	// copyFiles_for_results_Analysis_Later
	public static void copyFiles_for_results_Analysis_Later(
													String [] arr_top_N_files,
													String    destination_Folder
													){
		
		int c=0;
		String dest_="";
		try{
		
		while(c<arr_top_N_files.length){
			
			String source_=arr_top_N_files[c];
			System.out.println("arr_top_N_files[c]:"+c +" total:"+arr_top_N_files.length);
			
			String source_filename=new File(arr_top_N_files[c]).getName();
			
			 
			dest_=destination_Folder+source_filename;
			
			
			File source_file=new File(source_);
			File dest_file=new File(dest_);
			
			
			
			System.out.println("source_file:"+source_file+" dest_file:"+dest_file);
			FileUtils.copyFile(source_file, dest_file);
			c++;
		}
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	
	// main
	public static void main(String[] args) throws Exception {
		
		///this destination DOESNT need change..
		String temp_folder="ALL_files";
		//manual change		
		int flag=1;
		//manual change
		String intermediat_Folder="equ2";  //manual change  <<< SimpleLDA , equ1,equ2,equ3 
		// -----------------------------------------------------------------------------------------------------------------------------------
		// flag=1-> source is ds folder under "dummy.mergeall"..it is folder mentioned in var "baseFolder"
		// flag=2-> Switch source to destination and vice versa.. Source is folder mentioned in variable "destination_Folder",
		//			and Destination in var "baseFolder"
		// -----------------------------------------------------------------------------------------------------------------------------------
		
		String baseFolder="";
		String destination_Folder="";
			
		baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
		destination_Folder=baseFolder+"Result_Analysis/3rdAttempt/"+intermediat_Folder+"/"+temp_folder+"/";
		//
		if(flag==1){
				
		}
		else if(flag==2){
			//exchange SOURCE and DESTINATION
			String temp=destination_Folder;
			destination_Folder=baseFolder;
			baseFolder=temp;
		}
		
	    String []arr_top_N_files=new String[30];
	    // copy below files to runX folder
	    arr_top_N_files[0]=baseFolder+"Ground_Truth_final_1_OUT.txt_Top_N.txt";
	    arr_top_N_files[1]=baseFolder+"Ground_Truth_final_1_OUT.txt_poli_.txt_Top_N.txt";
	    arr_top_N_files[2]=baseFolder+"Ground_Truth_final_1_OUT.txt_trad_.txt_Top_N.txt";
	    
	    arr_top_N_files[3]=baseFolder+"Ground_Truth_final_2_OUT.txt_Top_N.txt";
	    arr_top_N_files[4]=baseFolder+"Ground_Truth_final_2_OUT.txt_poli_.txt_Top_N.txt";
	    arr_top_N_files[5]=baseFolder+"Ground_Truth_final_2_OUT.txt_trad_.txt_Top_N.txt";
	    
	    arr_top_N_files[6]=baseFolder+"Ground_Truth_final_3_OUT.txt_Top_N.txt";
	    arr_top_N_files[7]=baseFolder+"Ground_Truth_final_3_OUT.txt_poli_.txt_Top_N.txt";
	    arr_top_N_files[8]=baseFolder+"Ground_Truth_final_3_OUT.txt_trad_.txt_Top_N.txt";
	    
	    arr_top_N_files[9] =baseFolder+"Ground_Truth_final_4_OUT.txt_Top_N.txt";
	    arr_top_N_files[10]=baseFolder+"Ground_Truth_final_4_OUT.txt_poli_.txt_Top_N.txt";
	    arr_top_N_files[11]=baseFolder+"Ground_Truth_final_4_OUT.txt_trad_.txt_Top_N.txt";
	    
	    arr_top_N_files[12]=baseFolder+"Ground_Truth_final_5_OUT.txt_Top_N.txt";
	    arr_top_N_files[13]=baseFolder+"Ground_Truth_final_5_OUT.txt_poli_.txt_Top_N.txt";
	    arr_top_N_files[14]=baseFolder+"Ground_Truth_final_5_OUT.txt_trad_.txt_Top_N.txt";
	    
	    arr_top_N_files[15]=baseFolder+"Ground_Truth_final_6_OUT.txt_Top_N.txt";
	    arr_top_N_files[16]=baseFolder+"Ground_Truth_final_6_OUT.txt_poli_.txt_Top_N.txt";
	    arr_top_N_files[17]=baseFolder+"Ground_Truth_final_6_OUT.txt_trad_.txt_Top_N.txt";
	    
	    arr_top_N_files[18]=baseFolder+"Ground_Truth_final_7_OUT.txt_Top_N.txt";
	    arr_top_N_files[19]=baseFolder+"Ground_Truth_final_7_OUT.txt_poli_.txt_Top_N.txt";
	    arr_top_N_files[20]=baseFolder+"Ground_Truth_final_7_OUT.txt_trad_.txt_Top_N.txt";
	    
	    arr_top_N_files[21]=baseFolder+"tf_idf_sum_avg.txt";
	    arr_top_N_files[22]=baseFolder+"outputUniqNames_auth_DistinctDocs_no_duplicates.txt";
	    arr_top_N_files[23]=baseFolder+"Ground_Truth_OUT_auth_id_CUMM_SUM_AVG_tf_idf.txt";
	    arr_top_N_files[24]=baseFolder+"Ground_Truth_OUT_auth_id_CUMM_SUM_AVG_tf_idf_noDup.txt";
	    
	    //lda files
	    arr_top_N_files[25]=baseFolder+"LDA_d_by_z_prob.txt";
	    arr_top_N_files[26]=baseFolder+"LDA_d_by_z_prob_TITLE.txt";
	    arr_top_N_files[27]=baseFolder+"LDA_d_by_z_prob_AVG.txt";
		
	    
	    arr_top_N_files[28]=baseFolder+"Ground_Truth_final_8_OUT.txt_poli_.txt_Top_N.txt";
	    arr_top_N_files[29]=baseFolder+"Ground_Truth_final_8_OUT.txt_trad_.txt_Top_N.txt";
//	    arr_top_N_files[30]=baseFolder+"Ground_Truth_final_8_OUT.txt_Top_N.txt";
		
    	//copy_Top_N_files_to_runX_folder ( comment and uncomment )
		copyFiles_for_results_Analysis_Later(
    									arr_top_N_files,
    									destination_Folder
    									);
	}

}
