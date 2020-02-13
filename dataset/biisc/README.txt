===============================================================================
Database Usage Agreement
===============================================================================
By using the BII Sneeze-Cough Human Action Video Database, you accept the following database usage agreement:

1) Pleases cite the following reference in your work that makes any use of the database:

T. Thi, L. Cheng, L. Wang, N. Ye, J. Zhang, and S. Maurer-Stroh. Recognizing flu-like symptoms from videos. BMC Bioinformatics, 2014

2) Please use the videos/images for research purposes only.

3) If you reproduce videos/images in electronic or print media, please use ONLY those from the following subjects:

S001 S002 S003 S004 S006 S007 S008 S010 S011 S012 S013 S015 S016 S020


===============================================================================
Video File Naming Convention
===============================================================================
{subject id: 4 char}_{gender: 1 char}_{action: 4 char}_{stand or walk: 3 char}_{pose: 3 char}[_HF].avi
		{subject id}: Sxxx
		{gender}: M/F
		{action}:
			CALL: answer phone call
			COUG: cough
			DRIN: drink water
			SCRA: scratch head
			SNEE: sneeze
			STRE: stretch arms
			WAVE: wave hand
			WIPE: wipe glasses
		{stand or walk}: STD/WLK
		{pose}:
			FCE: face to the camera
			LFT: face to the left
			RGT: face to the right
		[_HF]: This suffix, if appears, denotes a horizontally flipped version of the original video.
Naming example:
	S001_M_SNEE_STD_FCE.avi
	S001_M_SNEE_STD_FCE_HF.avi


===============================================================================
Database Statistics
===============================================================================
Number of subjects: 20, M/F:12/8
Number of action types: 8, answer phone call, cough, drink water, scratch head, sneeze, stretch arms, wave hand, wipe glasses
Number of poses: 3, face to the camera, face to the left, face to the right
Number of locomotion types: 2, standing, walking
An extra horizontally flipped version has been synthesized for each of the videos.
Number of videos in total: 20x8x3x2x2=1920


===============================================================================
Video Format
===============================================================================
Frame rate: 10fps
Frame dimension: 480 x 290 pixels
Codec: FMP4
Length: around 4 seconds on average


===============================================================================
Data Split
===============================================================================
In our current experiments, the videos from subjects S002, S003, S004, S005, S006 are used for testing and the rest are used for training.
