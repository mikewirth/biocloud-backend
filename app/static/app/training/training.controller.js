'use strict';

angular.module('biocloud.training', ['biocloudRender', 'dndLists'])
  .controller('TrainingCtrl', function ($scope, $rootScope, Render) {
    console.log('stuff');
    // Render.render();

    $rootScope.transformations = [
        {
            "method": "gaussianBlur",
            "displayName": "Gaussian Blur",
            "parameters": {
                // "blrSize": 10,
            },
            "type": "transformation"
        },
        {
            "method": "noiseRemoval",
            "displayName": "Noise Removal",
            "parameters": {
                // "kSize": ""
            },
            "type": "transformation"
        },
        {
            "method": "filterBackgroundNoise",
            "displayName": "Filter Background Noise",
            "parameters": {
                // "subimageSize": 10,
                // "varianceTreshold": 10
            },
            "type": "transformation"
        },
        {
            "method": "removeHoles",
            "displayName": "Remove Holes",
            "parameters": {
                // "subimageSize": 10,
                // "varianceTreshold": 10
            },
            "type": "transformation"
        },
        {
            "method": "skeletonize",
            "displayName": "Skeletonize",
            "parameters": {},
            "type": "transformation"
        },
        {
            "method": "thresholding",
            "displayName": "Thresholding",
            "parameters": {},
            "type": "transformation"
        },
        {
            "method": "cellSegmentation",
            "displayName": "Cell Segmentation",
            "parameters": {},
            "type": "transformation"
        },
        {
            "method": "CropTool",
            "displayName": "Crop",
            "parameters": {
                "top": 5,
                "bottom": 5,
                "left": 5,
                "right": 5
            },
            "type": "transformation"
        },
        {
            "method": "edge_detection",
            "displayName": "Vessel Detection",
            "parameters": {
                // "subimageSize": 10,
                // "varianceTreshold": 10
            },
            "type": "transformation"
        },
    ];

    $rootScope.analysisBlocks = [
        {
            "method": "vesselWidth",
            "displayName": "Vessel Width",
            "parameters": {
                "pixelSize": 5,
            },
            "type": "analysis"
        },
    ];

                //     {
                //     "name": "blur",
                //     "type": "boolean",
                //     "default": "true"
                // }

    // $rootScope.renderingPipeline = {};
    if ($rootScope.renderingPipeline === undefined) {
        $rootScope.renderingPipeline = [];
    }


    $scope.refresh = function() {
        Render.render($rootScope.renderingPipeline, $rootScope.fields.selectedDataset);
    }

    $scope.deleteTransformation = function(index) {
        $rootScope.renderingPipeline.splice(index, 1);
    }
  });