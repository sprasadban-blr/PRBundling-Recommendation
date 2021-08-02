sap.ui.define([
	"sap/ui/core/mvc/Controller",
	'sap/ui/model/BindingMode',
	'sap/ui/model/json/JSONModel',
	'sap/viz/ui5/format/ChartFormatter',
	'sap/viz/ui5/api/env/Format',
	'./InitPage'
], function (Controller, BindingMode, JSONModel, ChartFormatter, Format, InitPageUtil) {
	"use strict";

	return Controller.extend("PRBUNDLE.PRBUNDLE.controller.data", {

		/**
		 * Called when a controller is instantiated and its View controls (if available) are already created.
		 * Can be used to modify the View before it is displayed, to bind event handlers and do other one-time initialization.
		 * @memberOf PRBUNDLE.PRBUNDLE.view.data
		 */
		onInit: function () {
				
				var oCard =  this.getView().byId('CardId');
				oCard.setVisible(false);
		},
		onButtonPress: function () {
			console.log("here");
			var data = [
				{
					"cluster": "cluster 1",
					"total": 1000
				},
				{
					"cluster": "cluster 2",
					"total": 1500
				},
				{
					"cluster": "cluster 3",
					"total": 400
				}
			];
			var oModel = new JSONModel();
			var dp1 = this.getView().byId("DP1").getValue().split("/");
			var fromdate = "20" +dp1[2] + (dp1[0].length==1? "0"+dp1[0] : dp1[0]) + (dp1[1].length==1? "0"+dp1[1] : dp1[1]);
			// var fromdate = (dp1[1].length==1? "0"+dp1[1] : dp1[1])  + (dp1[0].length==1? "0"+dp1[0] : dp1[0])  + "20" +dp1[2] ;
			var dp2 = this.getView().byId("DP2").getValue().split("/");
			var todate= "20" +dp2[2] + (dp2[0].length==1? "0"+dp2[0] : dp2[0]) + (dp2[1].length==1? "0"+dp2[1] : dp2[1])  ;
			// var todate = (dp2[1].length==1? "0"+dp2[1] : dp2[1]) + (dp2[0].length==1? "0"+dp2[0] : dp2[0])  + "20" +dp2[2] ;
			console.log(fromdate);
			console.log(todate);
			// oModel.setData(data);
			var ovizframe = this.getView().byId("idVizFrame1");
			$.ajax({
				type: "GET",
				dataType: "json",
				url: "http://127.0.0.1:5000/",
				async: false,
				data: {
					startdate : fromdate,
					enddate : todate,
				},
				contentType: "application/json",
				cors: true,
				secure: true,
				headers: {
					'Access-Control-Allow-Origin': '*'
				},
				success: function (data, textStatus, jqXHR) {
					oModel.setData(data);
					ovizframe.setModel(oModel,"data");
				}
			});
			
			
			// var ovizframe1 = this.getView().byId("idVizFrame2");
			// ovizframe1.setModel(oModel,"data");
			var oCard =  this.getView().byId('CardId');
			oCard.setVisible(true);
		},
		myChartClickHandler: function(oEvent){
			var content = oEvent.getParameters();
			var cluster = content.data[0].data.Cluster;
			var that = this;
			console.log(cluster);
			var oR = sap.ui.core.UIComponent.getRouterFor(this);
			var param = {
				clusterName : cluster
			}
			oR.navTo("detail",param);
		}

		/**
		 * Similar to onAfterRendering, but this hook is invoked before the controller's View is re-rendered
		 * (NOT before the first rendering! onInit() is used for that one!).
		 * @memberOf PRBUNDLE.PRBUNDLE.view.data
		 */
		//	onBeforeRendering: function() {
		//
		//	},

		/**
		 * Called when the View has been rendered (so its HTML is part of the document). Post-rendering manipulations of the HTML could be done here.
		 * This hook is the same one that SAPUI5 controls get after being rendered.
		 * @memberOf PRBUNDLE.PRBUNDLE.view.data
		 */
		//	onAfterRendering: function() {
		//
		//	},

		/**
		 * Called when the Controller is destroyed. Use this one to free resources and finalize activities.
		 * @memberOf PRBUNDLE.PRBUNDLE.view.data
		 */
		//	onExit: function() {
		//
		//	}

	});

});