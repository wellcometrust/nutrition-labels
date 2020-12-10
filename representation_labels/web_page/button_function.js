function button_active(c){
//show the right data set plot based on plot click

// loop through class 'boxhide' (all the plots and dataset text and hide them
	document.querySelectorAll('.box_hide').forEach(x => {
		x.style.display = "none";
	});

//get the right plot and text id name based on dataset name
	var text_name = c + '_text';
	var plot_name = c + '_plot';

// display the correct text and plots 
	document.getElementById(text_name).style.display = "block";
	document.getElementById(plot_name).style.display = "block";

// highlight the correct dataset tab button 
	document.querySelectorAll('.dataset_tab').forEach(y =>{
		if(y.id === c){
			y.style.backgroundColor = '#fec200';
		} else {
			y.style.backgroundColor = '#cb9b00';
		}
	})
	

	var rel_plot = plot_name + 'rel';
	
//	turning the right population tab yellow (when the page loads none of the plot elements have show)
//  so its only if the rel plot in each dataset_plot element == block is the rel plot shown.
	if(document.getElementById(rel_plot).style.display === "block"){
		document.getElementById('rel').style.backgroundColor = '#fec200';
		document.getElementById('no_rel').style.backgroundColor = '#cb9b00';
	} else {
		document.getElementById('no_rel').style.backgroundColor = '#fec200';
		document.getElementById('rel').style.backgroundColor = '#cb9b00';
	}
}

function plot_active(act,not_act){
// show correct populations plot based on tab click

	var on_id
	var off_id
	
// loop through dataset plot elements if it equals show then get the id of right populations plot

	document.querySelectorAll('.plot').forEach(x => {
		if(x.style.display === 'block'){
			on_id = x.id + act;
			off_id = x.id + not_act;
		}
	})

// show right population plot 
	document.getElementById(on_id).style.display = "block";
	document.getElementById(off_id).style.display = "none";
	
// change tab button colour
	document.getElementById(act).style.backgroundColor = '#fec200';
	document.getElementById(not_act).style.backgroundColor = '#cb9b00';

}