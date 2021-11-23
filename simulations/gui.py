import kivy
from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.config import Config
Config.set('graphics', 'resizable', '0') #0 being off 1 being on as in true/false
Config.set('graphics', 'width', '350')
Config.set('graphics', 'height', '250')
try:
    from src.utiliters.verilogBuilder import VerilogWithInitialization, VerilogWithoutInitialization
except ModuleNotFoundError:
    from utiliters.verilogBuilder import VerilogWithInitialization, VerilogWithoutInitialization

kivy.require('1.10.0')

__version__ = "0.1"

# achar unicode de acento hex(ord('ã')) dai usar \u00e3

class PaginaInicial(Screen):
    def start(self):
        # obtained through matrix H
        coefficient = [self.ids.txtH0.text, self.ids.txtH1.text, self.ids.txtH2.text, self.ids.txtH3.text, self.ids.txtH4.text, self.ids.txtH5.text, self.ids.txtH6.text]
        # LHC collision pattern
        pattern = self.ids.txtPattern.text
        # number of bits in the entrance of the algorithm
        algo = self.ids.spnAlgo.text
        # minimum iteration required, the real value is dependent of the pattern adopted
        iteration = int(self.ids.txtIterations.text)
        # if quantization still zero as above the precision above will be used
        quantization = int(self.ids.txtQuantization.text)
        # gain desirable to the simulation
        gain = int(self.ids.txtGain.text)
        # total number of windows to test
        lamb = float(self.ids.txtLambda.text)
        # path to work with
        path = './'
        #path = os.getcwd().replace('\\', '/') + '/../../../Verilog/Implementation/'

        verilog = VerilogWithoutInitialization([pattern, 10, iteration, gain, list(map(float, coefficient)), 0, path, quantization, algo, 2, lamb, 33])
        verilog.generate()
        print('Foi')


# class SegundaPagina(Screen):
#     pass


# class ScreenManagement(ScreenManager):
#     def switch_to_segundaPagina(self):
#         self.current = 'segundaPagina'
#
#     def switch_to_paginaInicial(self):
#         self.current = 'paginaInicial'


class guiApp(App):
    def build(self):
        self.title = 'Verilog Generator'
        self.root = PaginaInicial()
        return self.root


if __name__ == '__main__':
    guiApp().run()