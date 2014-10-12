'use strict';

angular.module('biocloud', ['ngAnimate', 'ngResource', 'ui.router', 'biocloud.training', 'biocloudRender'])
  .config(function ($stateProvider, $urlRouterProvider, $locationProvider) {
  	$urlRouterProvider.otherwise('/upload');

    // $locationProvider.html5Mode(true);

    $stateProvider
    .state('upload', {
      url: '/upload',
      templateUrl: 'app/upload/upload.html',
      controller: 'UploadCtrl'
    })
    .state('training', {
      url: '/train',
      templateUrl: 'app/training/training.html',
      controller: 'TrainingCtrl'
    })
    .state('batch', {
      url: '/batch',
      templateUrl: 'app/batch/batch.html',
      controller: 'BatchCtrl'
    });
  })
  .run(function($rootScope, Render) {

    $rootScope.fields = {
      "selectedDataset": ""
    };

    // $rootScope.batches = [
    //   {
    //     "name": "Colon cell reproduction",
    //     "id": "1"
    //   },
    //   {
    //     "name": "Retina 3",
    //     "id": "2"
    //   }
    // ];

    Render.images();

    $rootScope.batchImages = {
      "1":
        [
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180"
        ],
      "2":
        [
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180",
          "holder.js/100%x180"
        ]
    };
  });