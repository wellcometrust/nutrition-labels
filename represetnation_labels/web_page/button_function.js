function button_active(c){
	document.querySelectorAll('.box_hide').forEach(x => {
		x.style.display = "none";
	});
	var text_name = c + '_text';
	var plot_name = c + '_plot';
	document.getElementById(text_name).style.display = "block";
	document.getElementById(plot_name).style.display = "block";
	
	document.querySelectorAll('.dataset_tab').forEach(y =>{
		if(y.id === c){
			y.style.backgroundColor = '#fec200'
		} else {
			y.style.backgroundColor = '#cb9b00'
		}
	})
	
}

function plot_active(act,not_act){
	var on_id
	var off_id
	document.querySelectorAll('.plot').forEach(x => {
		if(x.style.display === 'block'){
			on_id = x.id + act;
			off_id = x.id + not_act;
		}
	})
	
	document.getElementById(on_id).style.display = "block";
	document.getElementById(off_id).style.display = "none";
	document.getElementById(act).style.backgroundColor = '#fec200'
	document.getElementById(not_act).style.backgroundColor = '#cb9b00'

}