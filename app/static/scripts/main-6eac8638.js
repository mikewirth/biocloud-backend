"use strict";angular.module("biocloud",["ngAnimate","ngResource","ui.router","biocloud.training","biocloudRender"]).config(["$stateProvider","$urlRouterProvider","$locationProvider",function(a,e){e.otherwise("/upload"),a.state("upload",{url:"/upload",templateUrl:"app/upload/upload.html",controller:"UploadCtrl"}).state("training",{url:"/train",templateUrl:"app/training/training.html",controller:"TrainingCtrl"}).state("batch",{url:"/batch",templateUrl:"app/batch/batch.html",controller:"BatchCtrl"})}]).run(["$rootScope","Render",function(a,e){a.fields={selectedDataset:""},e.images(),a.batchImages={1:["holder.js/100%x180","holder.js/100%x180","holder.js/100%x180","holder.js/100%x180"],2:["holder.js/100%x180","holder.js/100%x180","holder.js/100%x180","holder.js/100%x180","holder.js/100%x180","holder.js/100%x180","holder.js/100%x180","holder.js/100%x180","holder.js/100%x180","holder.js/100%x180","holder.js/100%x180","holder.js/100%x180"]}}]),angular.module("biocloud").controller("UploadCtrl",["$scope",function(){console.log("stuff")}]),angular.module("biocloud.training",["biocloudRender","dndLists"]).controller("TrainingCtrl",["$scope","$rootScope","Render",function(a,e,t){console.log("stuff"),e.transformations=[{method:"gaussianBlur",displayName:"Gaussian Blur",parameters:{},type:"transformation"},{method:"noiseRemoval",displayName:"Noise Removal",parameters:{},type:"transformation"},{method:"filterBackgroundNoise",displayName:"Filter Background Noise",parameters:{},type:"transformation"},{method:"removeHoles",displayName:"Remove Holes",parameters:{},type:"transformation"},{method:"skeletonize",displayName:"Skeletonize",parameters:{},type:"transformation"},{method:"thresholding",displayName:"Thresholding",parameters:{},type:"transformation"},{method:"cellSegmentation",displayName:"Cell Segmentation",parameters:{},type:"transformation"},{method:"CropTool",displayName:"Crop",parameters:{top:28,bottom:35,left:0,right:0},type:"transformation"},{method:"edge_detection",displayName:"Vessel Detection",parameters:{},type:"transformation"}],e.analysisBlocks=[{method:"vesselWidth",displayName:"Vessel Width",parameters:{pixelSize:1},type:"analysis"}],void 0===e.renderingPipeline&&(e.renderingPipeline=[]),a.refresh=function(){t.render(e.renderingPipeline,e.fields.selectedDataset)},a.deleteTransformation=function(a){e.renderingPipeline.splice(a,1)}}]),angular.module("biocloud.training").directive("resize",["$window",function(a){return function(e,t){var n=angular.element(a);e.getWindowDimensions=function(){return{h:n.height(),w:n.width()}},e.$watch(e.getWindowDimensions,function(a){e.windowHeight=a.h,e.windowWidth=a.w,t.css({height:a.h-140-50+"px"})},!0),n.bind("resize",function(){e.$apply()})}}]),angular.module("biocloud").controller("BatchCtrl",["$scope","$rootScope","Render",function(a,e,t){a.mode="start";console.log(a),a.startBatch=function(){t.batch(e.renderingPipeline,e.fields.selectedDataset).then(function(e){console.log(e),a.resultData=e.data,a.mode="results";var t=a.resultData.results.map(function(){return""}),n=a.resultData.results.map(function(a){return a.data.diameter}),e={labels:t,datasets:[{fillColor:"rgba(151,187,205,0.5)",strokeColor:"rgba(151,187,205,1)",pointColor:"rgba(151,187,205,1)",pointStrokeColor:"#fff",data:n}]};console.log(a);{var i=document.getElementById("testchart3").getContext("2d");new Chart(i).Line(e)}})},a.identifyBatch=function(a){return t.identifyBatch(a)}}]),angular.module("biocloudRender",[]).factory("Render",["$http","$rootScope",function(a,e){var t={};return t.render=function(e,n){var i={actions:e,selectedDataset:t.identifyBatch(n).id,showUntil:-1};a.post("http://ec2-54-72-186-15.eu-west-1.compute.amazonaws.com:5000/render",i,{responseType:"blob"}).success(function(a){var e=URL.createObjectURL(a);document.getElementById("renderImage").src=e}).error(function(){alert("There was an error rendering on the server.")})},t.batch=function(e,n){var i={actions:e,selectedDataset:t.identifyBatch(n).id},o=a.post("http://ec2-54-72-186-15.eu-west-1.compute.amazonaws.com:5000/batch",i).success(function(a){return a}).error(function(){alert("There was an error on the server.")});return o},t.images=function(){a.get("http://ec2-54-72-186-15.eu-west-1.compute.amazonaws.com:5000/imglist").success(function(a){e.batches=a.results,console.log(e.batches)}).error(function(){})},t.identifyBatch=function(a){for(var t in e.batches)if(e.batches[t].name==a)return e.batches[t]},t}]),function(a){try{a=angular.module("biocloud")}catch(e){a=angular.module("biocloud",[])}a.run(["$templateCache",function(a){a.put("components/topnavbar.html",'<div class="navbar navbar-inverse navbar-fixed-top" role="navigation"><div class="container-fluid"><div class="navbar-header"><button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target=".navbar-collapse"><span class="sr-only">Toggle navigation</span> <span class="icon-bar"></span> <span class="icon-bar"></span> <span class="icon-bar"></span></button> <a class="navbar-brand" href="#"><img class="logo" src="assets/logo.png" alt="BioCloud"></a></div><div class="navbar-collapse collapse"><ul class="nav navbar-nav navbar-right"><li><a ui-sref="upload" ui-sref-active="active">Upload images</a></li><li><a ui-sref="training" ui-sref-active="active">Train pipeline</a></li><li><a ui-sref="batch" ui-sref-active="active">Batch process</a></li><li><a href="#">Help</a></li></ul></div></div></div>')}])}(),function(a){try{a=angular.module("biocloud")}catch(e){a=angular.module("biocloud",[])}a.run(["$templateCache",function(a){a.put("components/navbar/navbar.html",'<nav class="navbar navbar-static-top navbar-inverse" ng-controller="NavbarCtrl"><div class="navbar-header"><a class="navbar-brand" href="https://github.com/Swiip/generator-gulp-angular"><span class="glyphicon glyphicon-home"></span> Gulp Angular</a></div><div class="collapse navbar-collapse" id="bs-example-navbar-collapse-6"><ul class="nav navbar-nav"><li class="active"><a ng-href="#">Home</a></li><li><a ng-href="#">About</a></li><li><a ng-href="#">Contact</a></li></ul><ul class="nav navbar-nav navbar-right"><li>Current date: {{ date | date:\'yyyy-MM-dd\' }}</li></ul></div></nav>')}])}(),function(a){try{a=angular.module("biocloud")}catch(e){a=angular.module("biocloud",[])}a.run(["$templateCache",function(a){a.put("app/batch/batch-images.html",'<div class="row"><div class="col-xs-6 col-md-3" ng-repeat="image in identifyBatch($parent.fields.selectedDataset).images track by $index"><a class="thumbnail"><img src="{{image}}" alt="..."></a></div></div>')}])}(),function(a){try{a=angular.module("biocloud")}catch(e){a=angular.module("biocloud",[])}a.run(["$templateCache",function(a){a.put("app/batch/batch-main.html",'<div class="imageview" resize=""><div class="center" ng-include="app/batch/batch-images.html"></div></div><div class="footer patternbg"><a ng-click="refresh()">Refresh image</a><ul class="renderingPipeline" dnd-list="renderingPipeline.transformations"><li ng-repeat="transformation in renderingPipeline.transformations" class="transformation{{renderingPipeline.showUntil == $index + 1 ? \' active\' : \'\'}}" dnd-draggable="transformation" dnd-effect-allowed="copyMove" dnd-moved="deleteTransformation($index)"><span>{{transformation.displayName}}</span></li></ul></div>')}])}(),function(a){try{a=angular.module("biocloud")}catch(e){a=angular.module("biocloud",[])}a.run(["$templateCache",function(a){a.put("app/batch/batch-results.html",'<div class="row"><div class="col-md-10"><h4 ng-if="mode == \'results\'">Average diameter over dataset: {{resultData.average.vesselWidth.diameter}}</h4><canvas id="testchart3" width="650" height="350"></canvas></div></div>')}])}(),function(a){try{a=angular.module("biocloud")}catch(e){a=angular.module("biocloud",[])}a.run(["$templateCache",function(a){a.put("app/batch/batch.html",'<div class="row batch"><div class="col-md-10"><h3>Select your training set:<select class="form-control" ng-model="fields.selectedDataset"><option value="">-</option><option ng-repeat="batch in batches">{{batch.name}}</option></select><a class="btn btn-primary" ng-click="startBatch()">Start batch evaluation</a></h3><div ng-if="mode == \'start\'" ng-include="\'app/batch/batch-images.html\'"></div><div ng-include="\'app/batch/batch-results.html\'"></div></div></div><div class="footer patternbg batch"><ul class="renderingPipeline" dnd-list="renderingPipeline"><li ng-repeat="transformation in renderingPipeline" class="transformation{{renderingPipeline.showUntil == $index + 1 ? \' active\' : \'\'}}" dnd-draggable="transformation" dnd-effect-allowed="copyMove" dnd-moved="deleteTransformation($index)"><span>{{transformation.displayName}}</span> <a ng-click="deleteTransformation($index)" class="delete"><i class="fa fa-times"></i></a></li></ul></div>')}])}(),function(a){try{a=angular.module("biocloud")}catch(e){a=angular.module("biocloud",[])}a.run(["$templateCache",function(a){a.put("app/training/training-main.html",'<div class="imageview" resize=""><div class="center"><img id="renderImage" onload="document.getElementById(\'emptyMessage\').style.display = \'none\'"><div id="emptyMessage" class="message"><span>Choose the steps of your processing pipeline and press the \'Refresh image\' button.</span></div></div></div><div class="footer patternbg"><div class="col-md-2 datasetSelector"><select class="form-control" ng-model="fields.selectedDataset"><option value="">Select dataset</option><option ng-repeat="batch in batches">{{batch.name}}</option></select><a class="btn btn-primary" ng-click="refresh()">Refresh image</a></div><div class="col-md-8 pipelineWrapper"><ul class="renderingPipeline" dnd-list="renderingPipeline"><li ng-repeat="transformation in renderingPipeline" class="transformation{{renderingPipeline.showUntil == $index + 1 ? \' active\' : \'\'}}" dnd-draggable="transformation" dnd-effect-allowed="copyMove" dnd-moved="deleteTransformation($index)"><span>{{transformation.displayName}}</span> <a ng-click="deleteTransformation($index)" class="delete"><i class="fa fa-times"></i></a></li></ul></div></div>')}])}(),function(a){try{a=angular.module("biocloud")}catch(e){a=angular.module("biocloud",[])}a.run(["$templateCache",function(a){a.put("app/training/training-sidebar.html",'<div class="sidebar patternbg"><h4>Image transformations</h4><ul class="transformationList clearfix"><li ng-repeat="transformation in transformations" class="transformation" dnd-draggable="transformation" dnd-effect-allowed="copy" dnd-copied=""><span>{{transformation.displayName}}</span></li></ul><h4>Analysis</h4><ul class="transformationList analysisBlocks"><li ng-repeat="transformation in analysisBlocks" class="transformation" dnd-draggable="transformation" dnd-effect-allowed="copy" dnd-copied=""><span>{{transformation.displayName}}</span></li></ul><div class="parameterPanel">Test</div></div>')}])}(),function(a){try{a=angular.module("biocloud")}catch(e){a=angular.module("biocloud",[])}a.run(["$templateCache",function(a){a.put("app/training/training.html",'<div class="row training"><ng-include src="\'app/training/training-sidebar.html\'"></ng-include><div class="col-sm-9 col-sm-offset-3 col-md-10 main" ng-include="\'app/training/training-main.html\'"></div></div>')}])}(),function(a){try{a=angular.module("biocloud")}catch(e){a=angular.module("biocloud",[])}a.run(["$templateCache",function(a){a.put("app/upload/upload.html",'<div class="jumbotron text-center"><h4>Upload your datasets</h4><input type="file" class="form-control"></div>')}])}();