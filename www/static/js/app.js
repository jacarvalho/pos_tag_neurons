var base_url = window.location.href;

// List of analysed POS tags
concepts = ['(', ')', ',', '.', 'CC', 'CD', 'DT', 'IN', 'JJ', 'NN', 'NNP', 'NNS', 'PRP', 'RB', 'RBR', 'RBS', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'SPACE', 'OTHER'];

concepts_grouped = ['(', ')', ',', '.', 'CC', 'CD', 'DT', 'IN', 'JJ', 'MD', 'NN', 'NNP', 'PRP', 'RB', 'TO', 'VB', 'SPACE', 'OTHER'];

// List of experiments for the logistic regression classifier
dataset_options = ['group_tags_250_lines', 'not_group_tags_250_lines', 'group_tags_500_lines', 'not_group_tags_500_lines', 'group_tags_nltk_data_1000', 'not_group_tags_nltk_data_1000'];


// Setting up after document loads
$( document ).ready(function() {
    // Setup datasets of the classifier
    dataset_options.forEach(function(dataset_option) { 
      // Set concept options in the page
      $("#dataset").append('<option value=\"' + dataset_option + '\">' + dataset_option + '</option>');
    });

    // Setup concepts to analyse
    concepts_grouped.forEach(function(concept) { 
      // Set concept options in the page
      $("#concept").append('<option value=\"' + concept + '\">' + concept + '</option>');
    });

    // Other setups
    sample_model();
    
    sample_reviews();
    classify_review();

    $("#dataset").val('group_tags_250_lines');
    $("#concept").val('VB');
    concept_neuron_lr_results();
    sample_concept_neuron();
    set_dataset();
});


// Rounds a value to decimals places
function round(value, decimals) {
  return Number(Math.round(value+'e'+decimals)+'e-'+decimals);
}


// Tranforms v ([-1, 1]) into a color between red and blue
function toColor(v) {
  // v is -1 to 1 initially
  if(v > 0) {
    var h = 200;
    var s = "60%";
    v = 1 - v; // invert so v = 0 is highest lightness (white)
    var l = (Math.floor(v * 40) + 60) + '%';
  } else {
    var h = 0;
    var s = "60%";
    v = -v;
    v = 1 - v; // invert too
    var l = (Math.floor(v * 40) + 60) + '%';
  }
  var s = sprintf('hsl(%d,%s,%s)', h,s,l);
  return s;
}


// Tranforms v ([0, 1]) into a color between red and blue
function toColor2(v) {
  // v is 0 to 1 initially
  var h = 200;
  var s = "60%";
  v = 1 - v; // invert so v = 0 is highest lightness (white)
  var l = (Math.floor(v * 40) + 60) + '%';
  var s = sprintf('hsl(%d,%s,%s)', h,s,l);
  return s;
}


// Samples the pre-trained language model
function sample_model() {
  // Set data for sampling
  var data = {};
  data["n_samples"] = parseInt($("#n_samples").val());
  if($('#sample_type_max_prob').is(':checked')) {
    data["sample_type"] = 0;
  } else {
    data["sample_type"] = 1;
  }
  data["temperature"] = parseFloat($("#temperature").val());
  data["prime"] = $("#prime_text").val();

  // console.log (data);

  // POST request to sample model
  $("#sample_model_result").html('');
  $.post(base_url + "sample", data, function(response) {
    // console.log (response);
    $("#sample_model_result").html(response.replace(/\n/g, "<br>"));
  });

}


// Set dataset
function set_dataset() {
  $("#concept").html('');

  var dataset = $("#dataset").val();

  if (dataset.includes('not_')) {
    // Setup concepts to analyse
    concepts.forEach(function(concept) { 
      // Set concept options in the page
      $("#concept").append('<option value=\"' + concept + '\">' + concept + '</option>');
    });  
  } else {
    // Setup concepts to analyse
    concepts_grouped.forEach(function(concept) { 
      // Set concept options in the page
      $("#concept").append('<option value=\"' + concept + '\">' + concept + '</option>');
    });  
  }

  var data = {};
  data["dataset"] = $("#dataset").val();
  // console.log (data);

  $.post(base_url + "set_dataset", data, function(response) {
    // console.log (response)
    table_results_html = response["table_results_html"];

    $("#table_results_classifier").html('');
    $("#table_results_classifier").html(table_results_html);

  });

}


// Displays pre computed logistic regression plots for concept neuron
function concept_neuron_lr_results() {
  var data = {};
  data["dataset"] = $("#dataset").val();
  data["concept"] = $("#concept").val();
  // console.log (data);

  // Set image sources after server response
  $.post(base_url + "concept_plots", data, function(response) {
    // console.log (response)
    lr_weights_fig_path = encodeURI(response["lr_weights"]);
    concept_neuron_hist_fig_path = encodeURI(response["concept_neuron_hist"]);
    
    // console.log (lr_weights_fig_path);
    // console.log (concept_neuron_hist_fig_path);

    $("#lr_weights_fig").attr("src", base_url + lr_weights_fig_path);
    $("#concept_neuron_fig").attr("src", base_url + concept_neuron_hist_fig_path);

  });
}


// Render text with colors of cell states
function render(id, concept, input_text, cell_states, pos_tag, render_probabilities=false) {
  $(id).html(''); // flush

  // Loop through each sentence
  for (k = 0; k < input_text.length; k++) {
    $(id).append('<br>');

    var break_after_n_chars = 30;
    var pos_tag_index = 0;
    var j = 0;

    var sentence_ = input_text[k];
    var cell_states_ = cell_states[k];
    var pos_tags_ = pos_tag[k];

    for(var i=0; i < sentence_.length; i++) {
      // Add character
      var letter = sentence_[i].toString();
      var cell_state = cell_states_[i];
      var cole;
      if (render_probabilities==true) {
        cole = toColor2(cell_state);
      } else {
        cole = toColor(Math.tanh(cell_state));
      }
      var css = 'background-color:' + cole;
      if (letter == ' ') {
        letter = '_';
        css += ';color:' + cole;
      }
      if (letter == '\n') {
        letter = '\\n';
      }
      $(id).append('<div class="d" style=' + css + '>' + letter + '</div>');

      // Add pos tag of character
      if ((i + 1) % break_after_n_chars == 0) {
        $(id).append('<br>');
        for (j=pos_tag_index; j <= i; j++) {
          // console.log (pos_tags_[j]);
          // console.log (concept);
          if (pos_tags_[j] === concept) {
            css_pos_tag = 'background-color:Orange;';
          } else {
            css_pos_tag = ';';
          }

          $(id).append('<div class="d_pos_tag" style=' + css_pos_tag + '>' + pos_tags_[j] + '</div>');

        }
        $(id).append('<br><br>');
        pos_tag_index += break_after_n_chars;
      }
    }

    // Add POS tags to first or last rows
    if (i > j) {
      $(id).append('<br>');
      for (var j=pos_tag_index; j < i; j++) {
        if (pos_tags_[j] === concept) {
          css_pos_tag = 'background-color:Orange;';
        } else {
          css_pos_tag = ';';
        }

        $(id).append('<div class="d_pos_tag" style=' + css_pos_tag + '>' + pos_tags_[j] + '</div>');
      }
      $(id).append('<br><br>');
    }
  }
}


// Render the text, colored with the cell state activation of the selected neuron
function sample_concept_neuron() {
  var data = {};
  data['input_text'] = $("#concept_text").val();
  data['neuron'] = $("#concept_neuron").val();
  data["dataset"] = $("#dataset").val();
  // console.log (data);

  $('#concept_text_vis').html(''); // flush
  $.post(base_url + "sample_concept_neuron", data, function(response) {
    // console.log (response);
    render('#concept_text_vis', $("#concept").val(), response['input_text'], response['cell_states'], response['pos_tag']);
  });
}


// Render the text, colored with the selected logistic regression results
function sample_concept_classifier() {
  var data = {};
  data['input_text'] = $("#concept_text").val();
  data["dataset"] = $("#dataset").val();
  data["concept"] = $("#concept").val();
  data["concept_classifier"] = $("#concept_classifier").val();

  // console.log (data);

  $('#concept_text_vis').html(''); // flush
  $.post(base_url + "sample_concept_classifier", data, function(response) {
    // console.log (response);
    render('#concept_text_vis', $("#concept").val(), response['input_text'], response['probabilities'], response['pos_tag'], render_probabilities=true);
  });
}


// Render reviews
function render_reviews(div, data, reviews=false) {
  $(div).html(''); // flush

  for(var i=0; i < data.length; i++) {
    var letter = data[i][0].toString();
    var probs = data[i][1];
    var cell_state = data[i][2];

    invert_color = 1;
    if (reviews == true) {
      invert_color = -1;
    }
    var cole = toColor(invert_color * Math.tanh(cell_state));
    css = 'background-color:' + cole;
    if (letter == ' ') {
      letter = '_';
      css += ';color:' + cole;
    }
    if (letter == '\n') {
      css += ';display:block;'
    }

    $(div).append('<div class="d_reviews" style=' + css + '>' + letter + '</div>');
  }
}



// Sample reviews
function sample_reviews() {
  var data = {};
  var div = "#sampling_reviews" ;
  // Reset text

  data['n_samples'] = $("#n_samples_reviews").val();

  if($('#sample_type_max_prob_reviews').is(':checked')) {
    data["sample_type"] = 0;
  } else {
    data["sample_type"] = 1;
  }
  data['temperature'] = $("#temperature_reviews").val();
  // console.log (data);

  $.post(base_url + "sample_reviews", data, function(result) {
    render_reviews(div + ' .vis', result, reviews=true);
  });
}


// Classify reviews
function classify_review() {
  var data = {};
  var div = "#classify_review" ;
  data['review_text'] = $("#review_text").val();

  $.post(base_url + "classify_review", data, function(result) {
    // console.log (result);
    tr_review_neuron = result["tr_review_neuron"];
    pred_probability = result["pred_probability"];

    var data = [];
    var review_text = $("#review_text").val();

    for (i = 0; i < review_text.length; i++) {
      data.push([review_text.charAt(i), 0, tr_review_neuron[i]]);
    }

    render_reviews(div + ' .vis', data, reviews=true);
    

    cell_state_neuron = round(tr_review_neuron[tr_review_neuron.length - 1], 3);
    
    // console.log (cell_state_neuron);
    max_sent = 1.30;
    if (cell_state_neuron < 0) {
      $("#sentiment_value_base").width((0.50 + 0.50/max_sent * cell_state_neuron) * $("#sentiment_bar").width());
      $("#sentiment_value_pos").width((-0.5 * cell_state_neuron/max_sent) * $("#sentiment_bar").width());
      $("#sentiment_value_pos").html(cell_state_neuron);
      $("#sentiment_value_neg").width(0);
      $("#sentiment_value_neg").html('');
    } else {
      $("#sentiment_value_base").width('50%');
      $("#sentiment_value_pos").width(0);
      $("#sentiment_value_pos").html('');
      $("#sentiment_value_neg").width(0.5 * cell_state_neuron/max_sent * $("#sentiment_bar").width());
      $("#sentiment_value_neg").html(cell_state_neuron);
    }
            
    pred_probability = round(pred_probability[0], 3);
    // console.log (pred_probability);
    $("#prob_value_pos").html((1 - pred_probability).toFixed(2));
    $("#prob_value_pos").width((1 - pred_probability) * $("#prob_bar").width());

    $("#prob_value_neg").html(pred_probability.toFixed(2));
    $("#prob_value_neg").width(pred_probability * $("#prob_bar").width());

  });
}
