sap.ui.define([
	"sap/ui/core/mvc/Controller"
], function (Controller) {
	"use strict";

	return Controller.extend("PRBUNDLE.PRBUNDLE.controller.root", {
		onInit: function () {

		},
		onItemSelect: function (oEvent) {
			var oItem = oEvent.getParameter("item");
			var sKey = oItem.getKey();
			if ((sKey === "data" || sKey === "analytics" )) {
				var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
				oRouter.navTo(sKey);

			} 
		}
	});
});