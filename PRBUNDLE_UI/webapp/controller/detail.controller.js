sap.ui.define([
	"sap/ui/core/mvc/Controller",
	'sap/ui/model/BindingMode',
	'sap/ui/model/json/JSONModel',
	'sap/viz/ui5/format/ChartFormatter',
	'sap/viz/ui5/api/env/Format',
	'./InitPage'
], function (Controller, BindingMode, JSONModel, ChartFormatter, Format, InitPageUtil) {
	"use strict";

	return Controller.extend("PRBUNDLE.PRBUNDLE.controller.detail", {

		/**
		 * Called when a controller is instantiated and its View controls (if available) are already created.
		 * Can be used to modify the View before it is displayed, to bind event handlers and do other one-time initialization.
		 * @memberOf PRBUNDLE.PRBUNDLE.view.detail
		 */
		onInit: function () {
			var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
			oRouter.getRoute("detail").attachPatternMatched(this._onObjectMatched, this);
		},
		_onObjectMatched: function (oEvent) {
			var limehtml =  this.getView().byId("limeBox");
			limehtml.setVisible(false);
			this.clustername = oEvent.getParameters("arguments").arguments.clusterName;
			console.log(this.clustername);
			this.getView().byId("p2").setTitle(this.clustername);
			var chartdata = [{
				"po": "PO 1",
				"total": 500
			}, {
				"po": "PO 2",
				"total": 600
			}, {
				"po": "PO 3",
				"total": 400
			}];
			var oModel = new JSONModel();

			// oModel.setData(chartdata);
			Format.numericFormatter(ChartFormatter.getInstance());
			var ovizframe = this.getView().byId("idVizFrame1");
			// ovizframe.setModel(oModel, "data");
			var ovizframe1 = this.getView().byId("idVizFrame2");
			// ovizframe1.setModel(oModel, "data");
			var that = this;
			var oTable = this.getView().byId("clusterDetail");
			$.ajax({
				type: "GET",
				dataType: "json",
				url: "http://127.0.0.1:5000/clusterdata",
				async: false,
				data: {
					cluster : that.clustername
				},
				contentType: "application/json",
				cors: true,
				secure: true,
				headers: {
					'Access-Control-Allow-Origin': '*'
				},
				success: function (data, textStatus, jqXHR) {
					var oTableModel = new JSONModel();
					oTableModel.setData(data['table']);
					oTable.setModel(oTableModel);
					console.log(oTableModel);
					oModel.setData(data['chart1']);
					ovizframe.setModel(oModel, "data");
					var ochart2Model =  new JSONModel();
					ochart2Model.setData(data['chart2']);
					ovizframe1.setModel(ochart2Model, "data");
					console.log(data);
				}
			});

		},
		handleLineItemPress: function(oEvent){
			
			var evn = oEvent.getSource().getBindingContext();
			var limehtml =  this.getView().byId("limeBox");
			// console.log(oEvent.getSource().getBindingContext().getProperty("BANFN"));
			var prData = {
				"MANDT" : evn.getProperty("MANDT"),
				"BANFN" : evn.getProperty("BANFN"),
				"BNFPO" : evn.getProperty("BNFPO"),
				"FLIEF" : evn.getProperty("FLIEF"),
				"EKORG" : evn.getProperty("EKORG"),
				"cluster" : evn.getProperty("cluster")
			};
			// console.log(data);
			var that = this;
			var html=this.getView().byId("html");
			$.ajax({
				type: "POST",
				dataType: "json",
				url: "http://127.0.0.1:5000/interpretResult",
				data: JSON.stringify(prData),
				contentType: "application/json",
				async: false,
				cors: true,
				secure: true,
				headers: {
					'Access-Control-Allow-Origin': '*'
				},
				success: function (data, textStatus, jqXHR) {
					if(!limehtml.getVisible()){
			        	console.log("done");
			        	limehtml.setVisible(true);
			        }
					console.log(data);
					that.rurl = "http://127.0.0.1:5000/loadInterpretaibility/" +  evn.getProperty("BANFN");
				    var data1 = "<iframe src='"+that.rurl+"' height='300px' width='1200px'/>";
			        html.setContent(data1);
			        
			        
					
				}
			});
			console.log("here");
			
			
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
		 * @memberOf PRBUNDLE.PRBUNDLE.view.detail
		 */
		//	onBeforeRendering: function() {
		//
		//	},

		/**
		 * Called when the View has been rendered (so its HTML is part of the document). Post-rendering manipulations of the HTML could be done here.
		 * This hook is the same one that SAPUI5 controls get after being rendered.
		 * @memberOf PRBUNDLE.PRBUNDLE.view.detail
		 */
		//	onAfterRendering: function() {
		//
		//	},

		/**
		 * Called when the Controller is destroyed. Use this one to free resources and finalize activities.
		 * @memberOf PRBUNDLE.PRBUNDLE.view.detail
		 */
		//	onExit: function() {
		//
		//	}

	});

});