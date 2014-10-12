'use strict';

angular.module('biocloud.training')
  .directive('resize', function ($window) {
    return function (scope, element) {
        var w = angular.element($window);
        scope.getWindowDimensions = function () {
            return {
                'h': w.height(),
                'w': w.width()
            };
        };
        scope.$watch(scope.getWindowDimensions, function (newValue, oldValue) {
            scope.windowHeight = newValue.h;
            scope.windowWidth = newValue.w;

            element.css({height: ((newValue.h - 140 - 50) + 'px')});
            /*
            scope.style = function () {
                console.log((newValue.h - 140) + 'px');
                return {
                    'height': (newValue.h - 140) + 'px'
                };
            };
            */

        }, true);

        w.bind('resize', function () {
            scope.$apply();
        });
    }
});