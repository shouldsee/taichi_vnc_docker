import hashlib

from server_util import add_url_params

import flask
from datetime import datetime
#import flask
def str2hex(x,encoding='utf-8'):
    return (hashlib.md5(x.encode(encoding)).hexdigest())

from apiflask import APIFlask, Schema, abort, APIBlueprint
from apiflask.fields import Integer, String
from apiflask.validators import Length, OneOf

from flask import url_for
url_for

app0 = APIFlask(__name__)
app = bp = APIBlueprint('root',__name__)

app0.config['PREFIX_GCAT'] = PREFIX_GCAT = prefix = '/gcat/'
pets = [
    {'id': 0, 'name': 'Kitty', 'category': 'cat'},
    {'id': 1, 'name': 'Coco', 'category': 'dog'}
]


class PetIn(Schema):
    name = String(required=True, validate=Length(0, 10))
    category = String(required=True, validate=OneOf(['dog', 'cat']))


class PetOut(Schema):
    id = Integer()
    name = String()
    category = String()



@app.get('/')
def say_hello():
    # returning a dict or list equals to use jsonify()
    return {'message': 'Hello this is root!'}


# @app.get('/pets/<int:pet_id>')
# @app.output(PetOut)
# def get_pet(pet_id):
#     '''
#     Document
#     '''
#     if pet_id > len(pets) - 1:
#         abort(404)
#     # you can also return an ORM/ODM model class instance directly
#     # APIFlask will serialize the object into JSON format
#     return pets[pet_id]


# @app.patch('/pets/<int:pet_id>')
# @app.input(PetIn(partial=True))
# @app.output(PetOut, description='The updated pet')
# @app.doc(tags=['Pet'])
# def update_pet(pet_id, data):
#     """Update a Pet
#     Update a pet with given data, the valid fields are `name` and `category`.
#     """    
#     if pet_id > len(pets) - 1:
#         abort(404)
#     for attr, value in data.items():
#         pets[pet_id][attr] = value
#     return pets[pet_id]

class SimpleMdMessage(Schema):
    '''
    `struct` is a path to a 
    '''
    struct = String(required=True, validate=Length(0, 150))
    md_steps = Integer()

class SimpleMdResult(Schema):
    result_pointer = String()

import json

import threading 

# from collections import defaultdict
# thread_dict=
tdict = {}

@app.get('/job_status/<string:action_name>/<string:shex>')
# @app.input(SimpleMdMessage)
def job_status(action_name, shex,json=json):
    tkey = action_name + '/' + shex
    data = tdict[tkey]
    t = data['thread']
    is_alive = t.is_alive()
    datajson = json.dumps(data,indent=2,default=lambda x:x.__repr__())
    return flask.render_template_string('''
    <html>
        <body>
        <p>

        Job submitted!

        When the calculation is done, you can access it at 

        {{datajson}}
        </p>
        <pre>
        </pre>
        <table border="1">
        <thead></thead>
        <tbody>

            <tr>
                <td>
                    is_alive
                </td>
                <td>
                    {{is_alive}}
                </td>
            </tr>


            <tr>
                <td>
                    ts_start
                </td>
                <td>
                    {{data["ts_start"]}}
                </td>
            </tr>

            <tr>
                <td>
                    ret
                </td>
                <td>
                    {{data["ret"]}}
                </td>
            </tr>

        </tbody>
        </table>
        </body>        
        </html>''', **locals())    

    #  data):

@app.post('/job_submit/<string:action_name>')
@app.input(SimpleMdMessage)
def job_submit(action_name, data):
    """
    submit a job to a targeted
    """
    data0 = data.copy()

    act = getattr(Actions,action_name)
    data['shex'] = str2hex(json.dumps(sorted(data.items())))
    tkey = f'''{action_name}/{data['shex']}'''

    def wrapper(data,tkey=tkey,act=act):
        try:
            tdict[tkey]['ret'] = act(data)
        except Exception as e:
            tdict[tkey]['error'] = e
            raise e

    t = threading.Thread(target=wrapper,args=(data,))
    tdict[tkey] = dict(thread=t, ts_start=datetime.now().isoformat(),ret=None,error=None)

    t.start()
    
    url_for_status = url_for( PREFIX_GCAT +".job_status",action_name=action_name, shex=data['shex'])
    # =tkey)
    dumps = json.dumps
    return flask.render_template_string('''
    <html>
        <body>
        <p>
        Job submitted!

        When the calculation is done, you can access it at 

        </p>
        <pre>
        </pre>
        <table border="1">
        <thead></thead>
        <tbody>

            <tr>
                <td>
                    recved data json
                </td>
                <td>
                    <pre>{{dumps(data0,indent=2)}}</pre>
                </td>
            </tr>


            <tr>
                <td>
                    job_status check url
                </td>
                <td>
                    <a href="{{url_for_status}}">{{url_for_status}}</a>
                    <iframe src="{{url_for_status}}"></iframe>
                </td>
            </tr>

        </tbody>
        </table>
        </body>
        
        </html>''', **locals())
        # locals())
        # data)




#### dynamic param in 
@app.post('/md_simple/')
@app.input(SimpleMdMessage)
#@app.output(PetOut, description='The updated pet')
#@app.doc(tags=['Pet'])
def _func(data):
    
    act = Actions.md_simple
    
    data['shex'] = str2hex(json.dumps(sorted(data.items())))
    ret = act(data)
    ret['datajson'] = json.dumps(data,indent=2)

    return flask.render_template_string('''
    <html>
        <body>
        {{rundir}}
        <pre>
    {{datajson}}
        </pre>
        <table border="1">
        <thead></thead>
        <tbody>
            <tr>
                <td>
                    url_ngl
                </td>
                <td>
                    <a href="{{url_ngl}}">{{url_ngl}}</a>
                </td>
            </tr>
        </tbody>
        </table>
        </body>
        
        </html>''', **ret)

class Actions(object):
    @staticmethod
    def md_simple(data):
        """
        Run a molecular simple dynamics simulation with gromacs
        Need to supply `struct`

        `TBC:DONE:impl a status checking page to show whether the long running process. probably with a term-like interface. or with PYPE.json` 
        `TBC:adding persistence mechanism to build index for existing tasks, probably with PYPE.json`


        The sole idea is that the parameter decides which function to call

        Option 1: clone and proxy the request 
        - Pros
            - proxy is useful tech
            - And yes I am making another request to a internal 
        - Cons
            - information may be lost during proxying
            - the same request will be parsed multiple times
        - work: 5/10
        Option 2: Refactor to get callable from the registry 
        - Pros
            - fast
        - Cons
            - not sure 



        <http://localhost:9001/mdsrv/webapp/?struc=file://static-prot/1PGB/5123456/md_0_1.gro&traj=file://static-prot/1PGB/5123456/md_0_1.xtc&data=/mdsrv/&mdsrv=/mdsrv/>
        <http://localhost:9001/mdsrv/webapp/?struc=/mdsrv/file/static-prot/1PGB/5123456/md_0_1.gro&traj=file://static-prot/1PGB/5123456/md_0_1.xtc&data=/mdsrv/&mdsrv=/mdsrv/>

        """    

        struct = data['struct']
        shex = data['shex']
        md_steps = data['md_steps']
        rundir = '/shared_data/md_simple/'    

        if struct.startswith('file://'):
            struct = struct[len('file://'):]
            assert os.path.exists(struct),struct
            # assert 
            raise NotImplementedError('The pype recipe is not compatible with file:// url ')
        elif struct.startswith('rcsb://'):
            struct = struct[len('rcsb://'):]
        else:
            assert NotImplementedError(struct)
        
        from know_simple_md import Controller,know_simple_md

        # md_steps=123455    
        pype2 = Controller.from_func(know_simple_md, 
            # GMX    = pype1.built["GMX"],
            GMX    = 'gmx',
            PDB_ID = struct,
            md_steps=md_steps,
            )    
        # outdir = rundir+'/'+struct + '/' + str(md_steps)
        outdir = rundir+'/'+shex

        pype2.run(rundir=outdir)


        data['shex'] = shex
        data['outdir'] = outdir
        reldir = outdir.lstrip('/')
        # fodir = 'file://'+outdir
        fodir = 'file://' + reldir
        fudir = f'http://localhost:9001/mdsrv/file/{outdir}'

        # data['url_ngl'] = f'http://localhost:9001/mdsrv/webapp/?struc={fudir}/md_0_1.gro&traj={outdir.lstrip("/")}/md_0_1.xtc&data=/mdsrv/&mdsrv=/mdsrv/'
        # data['url_ngl'] = f'http://localhost:9001/mdsrv/webapp/?struc={fodir}/md_0_1.gro&traj={fodir}/md_0_1.xtc&data=/mdsrv/&file=/mdsrv/'
        url_ngl = add_url_params( 'http://localhost:9001/mdsrv/webapp/',
            {
            "struc":f'{fodir}/md_0_1.gro',
            'traj':f'{fodir}/md_0_1.xtc',
            'data':'/mdsrv/',
            'file':'/mdsrv/'})

        data['url_ngl']  = url_ngl
        data['rundir']   = rundir
        return data


        # # return flask.render_template('filename.html')
        # return data





app = app0
# Swagger UI
app.config['SWAGGER_UI_CSS'] = 'https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.11.1/swagger-ui.min.css'
app.config['SWAGGER_UI_BUNDLE_JS'] = 'https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.11.1/swagger-ui-bundle.min.js'
app.config['SWAGGER_UI_STANDALONE_PRESET_JS'] = 'https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.11.1/swagger-ui-standalone-preset.min.js'
# Redoc
app.config['REDOC_STANDALONE_JS'] = 'https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js'
# Elements
app.config['ELEMENTS_JS'] = 'https://cdn.jsdelivr.net/npm/@stoplight/elements-dev-portal@1.7.4/web-components.min.js'
app.config['ELEMENTS_CSS'] = 'https://cdn.jsdelivr.net/npm/@stoplight/elements-dev-portal@1.7.4/styles.min.css'
# RapiDoc
app.config['RAPIDOC_JS'] = 'https://cdn.jsdelivr.net/npm/rapidoc@9.3.2/dist/rapidoc-min.min.js'
# RapiPDF
app.config['RAPIPDF_JS'] = 'https://cdn.jsdelivr.net/npm/rapipdf@2.2.1/src/rapipdf.min.js'



app.register_blueprint(bp,url_prefix=prefix,name=prefix)