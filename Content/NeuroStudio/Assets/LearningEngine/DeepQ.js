
// Copyright (C) 2020 Aaron Krumins - All Rights Reserved


const app = require('express')();
const http = require('http').Server(app);
const io = require('socket.io')(http);
const util = require('util');
const port = 3000;
const clients = [];


const tf = require('@tensorflow/tfjs');
const tfn = require("@tensorflow/tfjs-node");
// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
require('@tensorflow/tfjs-node');


//tf.enable_eager_execution();
//Server Web Client
app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

//When a client connects, bind each desired event to the client socket
io.on('connection', function(socket){
	//track connected clients via log
	clients.push(socket.id);
	clientConnectedMsg = 'User connected ' + util.inspect(socket.id) + ', total: ' + clients.length;
	//io.emit('chat message', clientConnectedMsg);
	console.log(clientConnectedMsg);

const model = tf.sequential();




	var	counter = 0;
	let n = 0;
	let m = 0;
	
	
	//track disconnected clients via log
	socket.on('disconnect', function(){
		clients.pop(socket.id);
		let clientDisconnectedMsg = 'User disconnected ' + util.inspect(socket.id) + ', total: ' + clients.length;
		io.emit('chat message', clientDisconnectedMsg);
		console.log(clientDisconnectedMsg);
		
		
		process.exit()
		
		
	})

	//multicast received message from client
	socket.on('chat message', function(msg){
		
	
	
		
	
		var jsonInput = JSON.parse(msg);
		
		var lastaction = jsonInput['lastaction'];

		var reward = jsonInput['reward'];
		var action = jsonInput['action'];
		var observations = jsonInput['observations'];
		var prevobservations = jsonInput['prevobservations'];
		var obsreducer = jsonInput['obsreducer'];
		var buffer = jsonInput['buffersize'];
		var batch_size = jsonInput['batchsize'];
		var learning_rate = jsonInput['learningrate'];
		var epochs = jsonInput['epochs'];
		var nodes = jsonInput['nodes'];
		var hiddenlayers = jsonInput['hiddenlayers'];
		var maxactions = jsonInput['maxactions'];
		var save_model = jsonInput['savemodel'];
		var load_model = jsonInput['loadmodel'];
		var modelname = jsonInput['modelname'];
		
		
observations = (observations / obsreducer);
prevobservations = (prevobservations / obsreducer);

		var evalflag = jsonInput['evalflag'];
		var gamma = jsonInput['gamma'];
		const gammaT =tf.tensor(gamma);
		const rewardT =tf.tensor(reward);
		
		var h = 0;
		
		//var epochs = 100;
		
			console.log("buffer size: " + buffer );		
		console.log("batchsize: " + batch_size);
		console.log("epochs: " + epochs);
		console.log("maxactions: " + maxactions);
		
		
if ( save_model  == 1 ){
	
	async function saveModel(){
	const saveResults = await model.save('file://./' + modelname);
	console.log('model saved');
}
saveModel();


} else if ( load_model  == 1 ){
	
	
//const model = tf.loadModel('file://./model-1a/model.json');
async function processModel(){

//if ( m == 0 ){	
const handler = tfn.io.fileSystem("./" + modelname + "/model.json");
const savedmodel = await tf.loadModel(handler);
//console.log(savedmodel);
 
		

	

console.log('The Saved Model Was Loaded');	

//prevobservations = 0.1011;
	replay=[];
				x_train = [];
					y_train = [];
		//init the grid matrix
for ( var i = 0; i < batch_size; i++ ) {
  replay[i] = [0,0,0,0,0];
}	
console.log("initialized grid matrix");		


	 

//console.log('new Model Successfully Compiled');	



		var	state = [0,0];
		
		var newstate = [0,0];
		var targetQval = [0,0,0,0];
		
		minibatch = [];
		
	
		state[0] = prevobservations;
		state[1] = lastaction;

       
		newstate[0] = observations; 
		newstate[1] = action; 
		
console.log("state" + state);
console.log("previous observations: " + prevobservations);
console.log("new state" + newstate);
console.log("observations: " + observations);
		if (reward !== 0) {
		console.log('I got a reward');}
		
		
		const state_tensor = tf.tensor2d([state]);
				
	qval = savedmodel.predict(state_tensor);
//predictedwinner = qval.argMax();
predictedwinner = tf.argMax(qval, 1);
var winnerSting =  predictedwinner.dataSync();

		
	io.emit('chat message', winnerSting);
	console.log('predicted winner: ' + winnerSting);




};

	processModel();
	
	//m = m + 1;
	process.on('unhandledRejection', error => {
  // Will print "unhandledRejection err is not defined"
  console.log('unhandledRejection', error.message);
});


process.removeAllListeners();



	
	
}


	else if ( load_model  != 1 ){
	
if ( n == 0 ){	
//const learningRate = 0.08;

 
     model.add(tf.layers.dense({
        inputShape: [2],
        units: nodes,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
	for ( var l = 1; l < hiddenlayers; l++ ) {
		  model.add(tf.layers.dense({
        units: nodes,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
	console.log("hidden layers: " + l);
	}

    model.add(tf.layers.dense({
        units:maxactions,
        kernelInitializer: 'VarianceScaling', 
        activation: 'linear'
    }));
 
 
 
 model.compile({loss: 'meanSquaredError', optimizer: tf.train.adam(learning_rate)});

console.log('old Model Successfully Compiled');			




				replay=[];
				x_train = [];
					y_train = [];


	
	//init the grid matrix
for ( var i = 0; i < batch_size; i++ ) {
  replay[i] = [0,0,0,0,0];
}	
console.log("initialized grid matrix");		
		
}	
	
	
	
	
		//if ( n == 0 ){	
		
		n = n + 1;
		
		console.log("n: " + n);
		
				

		//Batch Training
		
	var	state = [0,0];
		
		var newstate = [0,0];
		var targetQval = [0,0,0,0];
		
		minibatch = [];
		
	
		state[0] = prevobservations;
		state[1] = lastaction;

       
		newstate[0] = observations; 
		newstate[1] = action; 
		
console.log("state" + state);
console.log("previous observations: " + prevobservations);
console.log("new state" + newstate);
console.log("observations: " + observations);
		if (reward !== 0) {
console.log('I got a reward');

}	


		
			function getRandomSubarray(arr, size) {
    var shuffled = arr.slice(0), i = arr.length, min = i - size, temp, index;
    while (i-- > min) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
    }
    return shuffled.slice(min);
}
			

			if (counter < buffer ){

				
			replay[counter] = [state, action, reward, newstate];
		counter = counter + 1;	
			}
else {
minibatch = getRandomSubarray(replay, buffer);
console.log("action" + action);
for(let j = 0; j < batch_size; j++){

		const batchState = tf.tensor2d(minibatch[j][0],[1,2]);
		const batchNextState = tf.tensor2d(minibatch[j][3],[1,2]);

		
old_qval = model.predict(batchState);
newQ = model.predict(batchNextState);

maxQ = newQ.max();

maxQFloat = maxQ.get();

targetQval = old_qval;	


rewardBatch = minibatch[j][2];
actionBatch = minibatch[j][1];
const values = targetQval.dataSync();

update = (rewardBatch + (gamma * maxQFloat));

console.log('update: ' + update);


values[actionBatch] = update;
x_train[j] = batchState.dataSync();
y_train[j] = values;



} 

const xs = tf.tensor2d(x_train,[batch_size,2]);
const ys = tf.tensor2d(y_train,[batch_size,maxactions]);
console.log('buffer full');

console.log("xs" + xs);
console.log("ys" + ys);



model.fit(xs, ys, {
  epochs: epochs,
callbacks: {
    onEpochEnd: (epoch, log) => io.emit('logloss', `Epoch ${epoch}: loss = ${log.loss}`)
 }
});


counter = 0; }


const state_tensor = tf.tensor2d([state]);
				
qval = model.predict(state_tensor);
//predictedwinner = qval.argMax();
predictedwinner = tf.argMax(qval, 1);
var winnerSting =  predictedwinner.dataSync();

		
	io.emit('chat message', winnerSting);
	console.log('predicted winner: ' + winnerSting);

		
		};
		
		
	


		
		
	});
});

//Start the Server
http.listen(port, function(){
  console.log('listening on *:' + port);
 
});

// Copyright (C) 2020 Aaron Krumins - All Rights Reserved